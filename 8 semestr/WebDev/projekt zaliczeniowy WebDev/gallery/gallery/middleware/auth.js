const jwt = require("jsonwebtoken");
const user = require("../models/user");

module.exports = function (req, res, next) {
  if (req.user) {
    return next();
  }
  res.redirect('/users/login');
};