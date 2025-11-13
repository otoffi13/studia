const user = require("../models/user");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");

// GET - formularz logowania
exports.login_get = (req, res) => {
  res.render("login_form", { title: "Login" });
};

// POST - obsługa logowania
exports.login_post = async (req, res) => {
  const { username, password } = req.body;
  const foundUser = await user.findOne({ username }).exec();
  if (!foundUser) {
    return res.render("login_form", { title: "Login", messages: ["Invalid username or password"] });
  }
  const match = await bcrypt.compare(password, foundUser.password);
  if (!match) {
    return res.render("login_form", { title: "Login", messages: ["Invalid username or password"] });
  }
  // Tworzenie tokena JWT
  const token = jwt.sign(
    { userId: foundUser._id, username: foundUser.username },
    "sekretnyklucz", // w produkcji użyj zmiennej środowiskowej!
    { expiresIn: "2h" }
  );
  res.cookie("token", token, { httpOnly: true, sameSite: 'lax' });
  res.redirect("/");
};

// Wylogowanie
exports.logout = (req, res) => {
  res.clearCookie("token");
  res.redirect("/");
};