import express from "express";
import { addLog, getLogs } from "../controllers/logsController.js";

const router = express.Router();

router.post("/add", addLog);
router.get("/fetch", getLogs);

export default router;
