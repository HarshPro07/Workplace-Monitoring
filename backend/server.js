import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import { connectDB } from "./db/connect.js";
import logsRoutes from "./routes/logs.js";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

app.get("/health", (_, res) => res.json({ ok: true }));
app.use("/api/logs", logsRoutes);

const PORT = process.env.PORT || 3000;
connectDB();

// app.get("/", (req, res) => {
//   res.send("Workplace Monitoring Backend is running");
// });

app.listen(PORT, () => console.log(`🚀 Backend running on port ${PORT}`));
