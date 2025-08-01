import * as dotenv from 'dotenv';
dotenv.config();
import express from 'express';
import cors from 'cors';
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { GoogleGenAI } from "@google/genai";
import { Pinecone } from '@pinecone-database/pinecone';

const app = express();
app.use(cors({
  origin:["https://your-frontend.netlify.app", "http://localhost:3001"], // React frontend
  methods: ["GET", "POST"],
  allowedHeaders: ["Content-Type"]
}));
app.use(express.json()); // Handles JSON body

const ai = new GoogleGenAI({});

// ✅ Chat route with history passed from frontend
app.post("/chat", async (req, res) => {
  const { question, history } = req.body;

  try {
    // 📝 Rebuild history for AI
    const History = (history || []).map((msg) => ({
      role: msg.role,
      parts: [{ text: msg.content }]
    }));

    // 📝 Add latest user message
    History.push({
      role: 'user',
      parts: [{ text: question }]
    });

    // 🔄 Rewriting user query for clarity
    const rewrittenQuery = await ai.models.generateContent({
      model: "gemini-2.0-flash",
      contents: History,
      config: {
        systemInstruction: `You are a query rewriting expert. Rephrase the last user question as a standalone query.`,
      },
    });

    // 🧠 Generate embedding from rewritten query
    const embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GEMINI_API_KEY,
      model: 'text-embedding-004',
    });
    const queryVector = await embeddings.embedQuery(rewrittenQuery.text);

    // 🔍 Search Pinecone for context
    const pinecone = new Pinecone();
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);
    const searchResults = await pineconeIndex.query({
      topK: 10,
      vector: queryVector,
      includeMetadata: true,
    });

    // 📚 Build context
    const context = searchResults.matches
      .map(match => match.metadata.text)
      .join("\n\n---\n\n");

    // 🤖 Generate final response
    const response = await ai.models.generateContent({
      model: "gemini-2.0-flash",
      contents: [...History],
      config: {
        systemInstruction: `You are a Data Structures & Algorithms Expert.
        Use only the given context. If the answer is not found, say:
        "I could not find the answer in the provided document."

        Context: ${context}`,
      },
    });

    res.json({ answer: response.text });
  } catch (error) {
    console.error(error);
    res.status(500).json({ answer: "Something went wrong." });
  }
});

// 🚀 Start server
app.listen(5000, () => {
  console.log("✅ Server running on port 5000");
});
