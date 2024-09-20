"use client";
import { Box, Button, Stack, TextField } from "@mui/material";
import { Readex_Pro } from "next/font/google";
import Image from "next/image";
import { useState } from "react";


export default function Home() {
  const [messages, setMessages] = useState([{
    role: 'assistant',
    content: 'Hi! I am an AI agent here to assist. How can I help you today?'
  }]);

  const [message, setMessage] = useState('');

  const sendMessage = async () => {
    setMessage('')
    setMessages((messages)=> [
      ...messages,
      {role: "user", content: message},
      {role: "assistant", content: ''},
    ])
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify([...messages, {role: "user", content: message}]),
    }).then(async (res) => {
      const reader = res.body.getReader()
      const decoder = new TextDecoder()

      let results = ''
      return reader.read().then(async function processText({ done, value }) {
        if (done) {
          return results
        }
        const text = decoder.decode(value || new IntBArray(), { stream: true })
        setMessages((messages) => {
          let lastMessage = messages[messages.length - 1]
          let otherMessages = messages.slice(0, messages.length - 1)
          return [
            ...otherMessages,
            {
              ...lastMessage,
              content: lastMessage.content + text,
            },
          ]
        })
        return reader.read().then(processText)
      })
    })
  }


  return (
    
    <Box 
      width="100vw"
      height="100vh"
      display="flex"
      flexDirection="column"
      alignItems="center"
      justifyContent="center"
      >
        <Stack
          direction="column"
          width="600px"
          height="700px"
          border="1px solid black"
          p={2}
          spacing={3}
        >
          <Stack
            direction="column"
            spacing={2}
            flexGrow={1}
            overflow="auto"
            maxHeight="100%"
          >
            {
              messages.map((message, index) => (
                <Box 
                  key={index} 
                  display="flex" 
                  justifyContent={message.role === 'assistant' ? 'flex-start' : 'flex-end'}
                  alignItems="center"
                >
                  {message.role === 'assistant'}
                  <Box
                    bgcolor={message.role === 'assistant' ? '#FCD19C' : 'grey'}
                    color="black"
                    borderRadius={16}
                    p={3}
                    ml={message.role === 'assistant' ? 2 : 0}
                    mr={message.role === 'assistant' ? 0 : 2}
                  >
                    {message.content}
                  </Box>
                </Box>
              ))}
          </Stack>
          <Stack direction="row" spacing={2}>
            <TextField
              labell = "message"
              fullWidth
              value={message}
              onChange={(e) => setMessage(e.target.value)}
            />
            <Button variant="contained" style={{backgroundColor: '#000000'}} onClick={sendMessage}>Send</Button>
          </Stack>
        </Stack>
      </Box>
  )
}