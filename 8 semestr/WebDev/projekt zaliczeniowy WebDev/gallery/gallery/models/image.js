const mongoose = require("mongoose");

const Schema = mongoose.Schema;

const ImageSchema = new Schema({
  name: { type: String, required: true },
  description: String,
  filename: String, // <-- dodaj to pole!
  gallery: { type: Schema.Types.ObjectId, ref: "Gallery" },
  owner: { type: Schema.Types.ObjectId, ref: "User", required: true },
  date: { type: Date, default: Date.now },
});

module.exports = mongoose.model("Image", ImageSchema);
