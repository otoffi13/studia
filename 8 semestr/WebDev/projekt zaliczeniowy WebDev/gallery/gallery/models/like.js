const mongoose = require("mongoose");

const LikeSchema = new mongoose.Schema({
  user: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "User",
    required: true
  },
  image: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "Image",
    required: true
  },
  date: {
    type: Date,
    default: Date.now
  }
});

// Unikalny indeks - jeden użytkownik może polubić jedno zdjęcie tylko raz
LikeSchema.index({ user: 1, image: 1 }, { unique: true });

module.exports = mongoose.model("Like", LikeSchema); 