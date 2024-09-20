import { NextResponse } from "next/server"; 
import Groq from "groq-sdk";
import { Pinecone } from '@pinecone-database/pinecone';
import { HfInference } from '@huggingface/inference';
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { Document } from "langchain/document";

const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY
});


const inference = new HfInference(process.env.HUGGINGFACE_API_KEY); // Initialize HuggingFace

// Embed and upsert a document
async function upsertDocument(documentText, documentId) {
    
    const documentEmbedding = await inference.featureExtraction({
        model: "sentence-transformers/all-MiniLM-L6-v2",
        inputs: documentText
    });

    const upsertData = [
        {
            id: documentId,  // Document ID
            values: documentEmbedding,  // Embedding vector
            metadata: {
                content: documentText
            }
        }
    ];

    // Try and catch to deal with upsertion errors. 
    // Ideally want to upsert just once but I made it to assume that the document is subject to change.
    // But then need to take into account the older vectors. These are just thoughts and considerations.
    // All those considerations won't be implemented. Just enough to fulfill the prompt due to time constraints.

    try {
        const response = await pc.index("chatbot").namespace("webinfo").upsert(
            upsertData
        );
        // console.log("Upsert response:", response);
    } catch (error) {
        console.error("Error upserting document:", error);
    }
}

async function ensureIndexExists(indexName) {
    const indexes = await pc.listIndexes();
    if (Array.isArray(indexes) && !indexes.includes(indexName)) {
        console.log(`Index ${indexName} does not exist. Creating index...`);
        await pc.createIndex({
            name: indexName,
            dimension: 384, // Depends on embedding model
            metric: 'cosine', // Metric used for vector similarity
            spec: { 
                serverless: { 
                    cloud: 'aws', 
                    region: 'us-east-1' 
                }
            }
        });
        console.log(`Index ${indexName} created successfully.`);
    } else {
        console.log(`Index ${indexName} already exists.`);
    }
}

const systemPrompt = "Welcome! I'm here to assist you with any questions or issues you may have. How can I help you today?" 
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });


export async function POST(req) {
    // Go through RAG document, split it up and upsert
    const data = await req.json();
    const userQuery = data[data.length - 1].content;
    const fs = require('fs').promises;
    const path = process.env.FILE_PATH;
    
    const documentText = await fs.readFile(path, 'utf8');
    const docs = [new Document({ pageContent: documentText })];

    console.log("Splitting document text...");
    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 300,
        chunkOverlap: 1,
        separators: ["."],
    });

    await ensureIndexExists('chatbot')

    const allSplits = await textSplitter.splitDocuments(docs);
    console.log("Total splits:", allSplits.length);


    console.log("Upserting document splits...");

    if (allSplits.length > 0) {
        await Promise.all(allSplits.map((split, i) => upsertDocument(split.pageContent, `doc-${i+1}`)));
    } else {
        console.warn("No splits were generated from the document text.");
    }


    // Step 1: Embed the user's query
    const queryEmbedding = await inference.featureExtraction({
        model: 'sentence-transformers/all-MiniLM-L6-v2',
        inputs: userQuery
    });
    console.log("User Query: ", userQuery)

  

    // Step 2: Retrieve relevant documents from Pinecone
    const retrievalResponse = await pc.index('chatbot').namespace('webinfo').query({
        topK: 1,
        vector: queryEmbedding,
        includeMetadata: true,
        includeValues: true,
    });

    const relevantDocs = retrievalResponse.matches.map(match => match.metadata.content).join(' ');
    console.log("Relevant documents:", relevantDocs);

    // Step 3: Pass the relevant documents and user's query to the generative model
    const completion = await groq.chat.completions.create({
        messages: [
            {
                role: "system",
                content: systemPrompt
            },
            ...data,
            {
                role: "assistant",
                content: `Base your answers ONLY off of the relevant information and NOTHING else. Relevant Information: ${relevantDocs}`
            },
            {
                role: 'user',
                content: userQuery 
            },
        ],
        model: "llama3-8b-8192", // or another appropriate Groq model
        stream: true,
        max_tokens: 500,
    });

    console.log("LLM response before stream: ", completion)

    const stream = new ReadableStream({
        async start(controller) {
            const encoder = new TextEncoder()
            try {
                for await (const chunk of completion) {
                    const content = chunk.choices[0].delta?.content
                    if (content) {
                        const text = encoder.encode(content)
                        controller.enqueue(text)
                    }
                }
            }
            catch (error) {
                controller.error(error)
            } finally {
                controller.close()
            }
        }
    })

    console.log("LLM response after stream: ", stream)

    return new NextResponse(stream)
}