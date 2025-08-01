import * as dotenv from 'dotenv';
dotenv.config();
import express from 'express';
import cors from 'cors';
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { GoogleGenAI } from "@google/genai";
import { Pinecone } from '@pinecone-database/pinecone';

const app = express();

app.use(cors({
  origin: [
    "https://meek-fox-ee45ee.netlify.app", // replace with your real Netlify frontend
    "http://localhost:3000"
  ],
  methods: ["GET", "POST"],
  allowedHeaders: ["Content-Type"]
}));

app.use(express.json()); // parse JSON body

const ai = new GoogleGenAI({});

app.post("/chat", async (req, res) => {
  const { question, history } = req.body;

  try {
    const History = (history || []).map((msg) => ({
      role: msg.role,
      parts: [{ text: msg.content }]
    }));

    History.push({
      role: 'user',
      parts: [{ text: question }]
    });

    // Rewriting user query
    const rewritten = await ai.models.generateContent({
      model: "gemini-2.0-flash",
      contents: History,
      config: {
        systemInstruction: `Rephrase the last user question as a standalone query.`,
      },
    });

    const rewrittenQuery = rewritten?.text || question;

    // Embedding
    const embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GEMINI_API_KEY,
      model: 'text-embedding-004',
    });
    const queryVector = await embeddings.embedQuery(rewrittenQuery);

    // Pinecone search
    const pinecone = new Pinecone();
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

    const searchResults = await pineconeIndex.query({
      topK: 10,
      vector: queryVector,
      includeMetadata: true,
    });

    const context = searchResults.matches
      .map(match => match.metadata.text)
      .join("\n\n---\n\n");

    // Generate final response
    const response = await ai.models.generateContent({
      model: "gemini-2.0-flash",
      contents: [...History],
      config: {
        systemInstruction: `Use only the given context. If the answer is not found, say: "I could not find the answer in the provided document." Context: ${context}`,
      },
    });

    res.json({ answer: response.text });
  } catch (error) {
    console.error(error);
    res.status(500).json({ answer: "Something went wrong." });
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`✅ Server running on port ${PORT}`);
});
