var createError = require('http-errors');
var express = require('express');
var path = require('path');
var cookieParser = require('cookie-parser');
var logger = require('morgan');
const jwt = require("jsonwebtoken");
const userModel = require("./models/user");

// import routers - zadania temat8
var indexRouter = require('./routes/index');
var usersRouter = require('./routes/users');

// import routers - zadania temat9
var galleriesRouter = require('./routes/galleries');
var imagesRouter = require('./routes/images');

// app object
var app = express();

// view engine setup
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'pug');

app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));

// Middleware JWT - ustawia req.user jeśli token jest poprawny
app.use(async (req, res, next) => {
  const token = req.cookies.token;
  if (token) {
    try {
      const decoded = jwt.verify(token, "sekretnyklucz");
      req.user = await userModel.findById(decoded.userId).exec();
    } catch (err) {
      req.user = null;
    }
  } else {
    req.user = null;
  }
  next();
});

app.use((req, res, next) => {
  res.locals.user = req.user;
  next();
});

// use routers - zadania temat8
app.use('/', indexRouter);
app.use('/users', usersRouter);

// use routers - zadania temat9
app.use('/galleries', galleriesRouter);
app.use('/images', imagesRouter);

// catch 404 and forward to error handler
app.use(function(req, res, next) {
  next(createError(404));
});

// error handler
app.use(function(err, req, res, next) {
  // set locals, only providing error in development
  res.locals.message = err.message;
  res.locals.error = req.app.get('env') === 'development' ? err : {};

  // render the error page
  res.status(err.status || 500);
  res.render('error');
});

// MongoDB/Mongoose connection setup...
const mongoose = require("mongoose");

mongoose.set("strictQuery", false);

const mongoDB = "mongodb://localhost:27017/GalleryDB";

async function main() {
  await mongoose.connect(mongoDB);
}
main().catch((err) => console.log(err));

// ...MongoDB/Mongoose connection setup

module.exports = app;
