import Log from "../models/Log.js";

export const addLog = async (req, res) => {
  try {
    const log = await Log.create(req.body);
    res.status(201).json({ success: true, data: log });
  } catch (err) {
    res.status(400).json({ success: false, error: err.message });
  }
};

export const getLogs = async (_, res) => {
  try {
    const logs = await Log.find().sort({ timestamp: -1 }).limit(50);
    res.status(200).json({ success: true, data: logs });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};
