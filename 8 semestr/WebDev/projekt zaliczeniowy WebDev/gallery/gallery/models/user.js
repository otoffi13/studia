const mongoose = require("mongoose");

const Schema = mongoose.Schema;

const UserSchema = new Schema({
  name: { type: String, maxLength: 100 },
  surname: { type: String, maxLength: 100 },
  username: { type: String, maxLength: 100 },
  password: { type: String, required: true }, // dodane pole
  date: { type: Date, default: Date.now }, // data utworzenia konta
}, { collection: 'users'});

// Export model
module.exports = mongoose.model("User", UserSchema);

