import mongoose from "mongoose";

const logSchema = new mongoose.Schema({
  timestamp: {
    type: Date,
    default: Date.now,
    index: true, // for sorting and analysis later
  },
  yawn: { type: Boolean, default: false },
  drowsy: { type: Boolean, default: false },
  tilt: { type: Boolean, default: false },
  phone: { type: Boolean, default: false },
  fatigue_score: { type: Number, default: 0 },
  productivity_score: { type: Number, default: 100 },
});

export default mongoose.model("Log", logSchema);
