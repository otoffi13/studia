const user = require("../models/user");
const Gallery = require("../models/gallery");
const Image = require("../models/image");
const Comment = require("../models/comment");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");

// asynchronicznie
const asyncHandler = require("express-async-handler");

// NEW USER

// GET - Kontroler wyświetlania formularza dodawania nowego usera (metoda GET).
exports.user_add_get = function(req, res) {
  res.render('user_form', { title: 'Zarejestruj się' });
};

// Import funkcji walidatora.
const { body, validationResult } = require("express-validator");

// POST - Kontroler (a właściwie lista kontrolerów) obsługi danych z formularza dodawania nowego usera (metoda POST).
exports.user_add_post = [
  // Walidacja i sanityzacja danych z formularza.
  body("name")
    .trim()
    .isLength({ min: 2 })
    .escape()
    .withMessage("Imię jest za krótkie."),

  body("surname")
    .trim()
    .isLength({ min: 2 })
    .escape()
    .withMessage("Nazwisko jest za krótkie."),

  body("username", "Nazwa użytkownika musi zawierać co najmniej 3 znaki")
    .trim()
    .isLength({ min: 3 })
    .escape()
    .withMessage("Nazwa użytkownika musi mieć co najmniej 3 znaki."),

  body("password", "Hasło jest za krótkie!")
    .isLength({ min: 8 }),

  // Przetwarzanie danych po walidacji i sanityzacji
  asyncHandler(async (req, res, next) => {
    // Pozyskanie z request obiektu błędu i jego ewentualna obsługa.
    const errors = validationResult(req);

    if (!req.body.password) {
      return res.render("user_form", {
        title: "Dodaj użytkownika:",
        user: req.body,
        messages: ["Hasło jest wymagane"]
      });
    }

    // Zaszyfrowanie hasła
    const passwordHash = await bcrypt.hash(req.body.password, 10);

    // Tworzenie obiektu/dokumentu newuser z modelu User po 'oczyszczeniu' danych
    const newuser = new user({
      name: req.body.name,
      surname: req.body.surname,
      username: req.body.username,
      password: passwordHash,
      date: new Date(),
    });
  
    if (!errors.isEmpty()) {
      // Jeśli pojawiły się błędy - ponownie wyrenderuj formularz i wypełnij pola 
      // wprowadzonymi wcześniej danymi ora komunikatami błędów 
      // Roboczej tablica komunikatów:
  
      let myMessages=[]
      errors.array().forEach(err => myMessages.push(err.msg))

      res.render("user_form", {
        title: "Dodaj użytkownika:",
        user: newuser,
        messages: myMessages,  
      });
      return;
    }
    // Dane z formularza są poprawne.
    // Należy jeszcze sprawdzić czy w bazie istnieje już użytkownik 
    // o tym samym username
    const userExists = await user.findOne({ username: req.body.username })
      .collation({ locale: "pl", strength: 2 })
      .exec();

    if (userExists) {
      // Błąd - użytkownik już istnieje w bazie - ponownie wyrenderuj formularz, wypełnij pola 
      // wprowadzonymi wcześniej danymi, wydrukuj błąd

      res.render("user_form", {
      title: "Dodaj użytkownika:",
      user: newuser,
      messages: [`Nazwa użytkownika "${newuser.username}" już istnieje!`]
      });
      return;
    }
      
    // Zapisz do bazy nowego użytkownika.
    // Wyświetl pusty formularz i komunikat.
    await newuser.save()
      .then(res.render("user_form", {
      title: "Dodaj użytkownika:",
      user: {},
      messages: [`Użytkownik "${newuser.username}" został dodany`]
      }))
  }),

];

// GET - formularz logowania
exports.user_login_get = (req, res) => {
  res.render("login_form", { title: "Logowanie użytkownika" });
};

// POST - Kontroler przetwarzania formularza logowania
exports.user_login_post = (req, res, next) => {
  let username = req.body.username;
  let password = req.body.password;

  user.findOne({ username })
    .then((foundUser) => {
      if (foundUser) {
        bcrypt.compare(password, foundUser.password, function (err, result) {
          if (err) {
            return res.render("login_form", { title: "Logowanie użytkownika", messages: ["Błąd szyfrowania!"] });
          }
          if (result) {
            // Login OK - generujemy token JWT i zapisujemy w cookie
            let token = jwt.sign(
              { userId: foundUser._id, username: foundUser.username },
              'sekretnyklucz', // użyj tego samego klucza co w middleware!
              { expiresIn: '2h' }
            );
            res.cookie('token', token, { httpOnly: true, sameSite: 'lax' });
            res.redirect('/');
          } else {
            // Złe hasło
            res.render("login_form", { title: "Logowanie użytkownika", messages: ["Nieprawidłowe hasło!"] });
          }
        });
      } else {
        // Nie znaleziono użytkownika
        res.render("login_form", { title: "Logowanie użytkownika", messages: ["Nie znaleziono użytkownika!"] });
      }
    })
    .catch((err) => {
      res.render("login_form", { title: "Logowanie użytkownika", messages: ["Błąd: " + err.message] });
    });
};

// GET - Kontroler wylogowania
exports.user_logout_get = (req, res, next) => {
  res.clearCookie('token');
  res.redirect('/');
};

const User = require('../models/user');

// Usuwanie użytkownika (tylko admin)
exports.user_delete_post = asyncHandler(async (req, res, next) => {
  if (!req.user || req.user.username !== 'admin') {
    return res.status(403).send('Dostęp zabroniony');
  }
  await User.findByIdAndDelete(req.params.id);
  res.redirect('/users');
});

// GET - Lista użytkowników (dla wszystkich zalogowanych użytkowników)
exports.user_list = async function(req, res) {
  if (!req.user) {
    return res.redirect('/users/login');
  }
  
  const searchQuery = req.query.search || '';
  let users;
  
  if (searchQuery) {
    // Search by username (case-insensitive)
    users = await user.find({ 
      username: { $regex: searchQuery, $options: 'i' } 
    }).exec();
  } else {
    // Get all users
    users = await user.find().exec();
  }
  
  // Pobierz statystyki dla każdego użytkownika
  const usersWithStats = await Promise.all(users.map(async (u) => {
    const galleries = await Gallery.find({ user: u._id }).countDocuments();
    const images = await Image.find({ owner: u._id }).countDocuments();
    return {
      ...u.toObject(),
      date: u.date || new Date(), // fallback dla istniejących użytkowników bez daty
      stats: { galleries, images }
    };
  }));
  
  res.render('user_list', { 
    title: 'Użytkownicy', 
    users: usersWithStats,
    currentUser: req.user,
    searchQuery: searchQuery
  });
};

// GET - Profil użytkownika
exports.user_profile_get = asyncHandler(async (req, res, next) => {
  if (!req.user) {
    return res.redirect('/users/login');
  }
  
  // Pobierz statystyki użytkownika
  const userGalleries = await Gallery.find({ user: req.user._id }).exec();
  const userImages = await Image.find({ owner: req.user._id }).exec();
  
  res.render('user_profile', {
    title: 'Moje konto',
    user: req.user,
    stats: {
      galleries: userGalleries.length,
      images: userImages.length
    }
  });
});

// POST - Aktualizacja profilu użytkownika
exports.user_profile_post = [
  body("name")
    .trim()
    .isLength({ min: 2 })
    .escape()
    .withMessage("Imię musi mieć co najmniej 2 znaki."),

  body("surname")
    .trim()
    .isLength({ min: 2 })
    .escape()
    .withMessage("Nazwisko musi mieć co najmniej 2 znaki."),

  body("username", "Nazwa użytkownika musi zawierać co najmniej 3 znaki")
    .trim()
    .isLength({ min: 3 })
    .escape()
    .withMessage("Nazwa użytkownika musi mieć co najmniej 3 znaki."),

  asyncHandler(async (req, res, next) => {
    if (!req.user) {
      return res.redirect('/users/login');
    }

    const errors = validationResult(req);
    
    // Sprawdź, czy nowa nazwa użytkownika nie jest już zajęta (jeśli się zmieniła)
    if (req.body.username !== req.user.username) {
      const userExists = await user.findOne({ username: req.body.username })
        .collation({ locale: "pl", strength: 2 })
        .exec();
      
      if (userExists) {
        return res.render('user_profile', {
          title: 'Moje konto',
          user: req.user,
          messages: [`Nazwa użytkownika "${req.body.username}" jest już zajęta!`],
          stats: {
            galleries: await Gallery.find({ user: req.user._id }).countDocuments(),
            images: await Image.find({ owner: req.user._id }).countDocuments()
          }
        });
      }
    }

    if (!errors.isEmpty()) {
      return res.render('user_profile', {
        title: 'Moje konto',
        user: req.user,
        messages: errors.array().map(e => e.msg),
        stats: {
          galleries: await Gallery.find({ user: req.user._id }).countDocuments(),
          images: await Image.find({ owner: req.user._id }).countDocuments()
        }
      });
    }

    // Zaktualizuj dane użytkownika
    await user.findByIdAndUpdate(req.user._id, {
      name: req.body.name,
      surname: req.body.surname,
      username: req.body.username
    });

    // Pobierz zaktualizowane dane użytkownika
    const updatedUser = await user.findById(req.user._id);
    
    res.render('user_profile', {
      title: 'Moje konto',
      user: updatedUser,
      messages: ['Dane konta zostały zaktualizowane!'],
      stats: {
        galleries: await Gallery.find({ user: req.user._id }).countDocuments(),
        images: await Image.find({ owner: req.user._id }).countDocuments()
      }
    });
  })
];

// GET - Strona usuwania konta
exports.user_delete_account_get = asyncHandler(async (req, res, next) => {
  if (!req.user) {
    return res.redirect('/users/login');
  }
  
  // Pobierz statystyki użytkownika
  const userGalleries = await Gallery.find({ user: req.user._id }).exec();
  const userImages = await Image.find({ owner: req.user._id }).exec();
  
  res.render('user_delete_account', {
    title: 'Usuń konto',
    user: req.user,
    stats: {
      galleries: userGalleries.length,
      images: userImages.length
    }
  });
});

// POST - Usuwanie konta
exports.user_delete_account_post = asyncHandler(async (req, res, next) => {
  if (!req.user) {
    return res.redirect('/users/login');
  }

  // Admin nie może usunąć swojego konta
  if (req.user.username === 'admin') {
    return res.render('user_delete_account', {
      title: 'Usuń konto',
      user: req.user,
      messages: ['Administrator nie może usunąć swojego konta. Skontaktuj się z innym administratorem.'],
      stats: {
        galleries: await Gallery.find({ user: req.user._id }).countDocuments(),
        images: await Image.find({ owner: req.user._id }).countDocuments()
      }
    });
  }

  // Sprawdź, czy użytkownik potwierdził usunięcie
  if (req.body.confirm !== 'true') {
    return res.redirect('/users/delete-account');
  }

  try {
    // Usuń wszystkie zdjęcia użytkownika
    const userImages = await Image.find({ owner: req.user._id });
    for (const img of userImages) {
      // Usuń plik fizyczny
      const fs = require('fs');
      const path = require('path');
      const filePath = path.join(__dirname, '../public/images', img.filename);
      if (fs.existsSync(filePath)) {
        fs.unlinkSync(filePath);
      }
    }
    await Image.deleteMany({ owner: req.user._id });

    // Usuń wszystkie galerie użytkownika
    await Gallery.deleteMany({ user: req.user._id });

    // Usuń wszystkie komentarze użytkownika
    await Comment.deleteMany({ author: req.user._id });

    // Usuń konto użytkownika
    await user.findByIdAndDelete(req.user._id);

    // Wyloguj użytkownika
    res.clearCookie('token');
    
    res.render('user_deleted', {
      title: 'Konto usunięte',
      messages: ['Twoje konto zostało pomyślnie usunięte wraz ze wszystkimi galeriami i zdjęciami.']
    });
  } catch (error) {
    console.error('Błąd podczas usuwania konta:', error);
    res.render('user_delete_account', {
      title: 'Usuń konto',
      user: req.user,
      messages: ['Wystąpił błąd podczas usuwania konta. Spróbuj ponownie.'],
      stats: {
        galleries: await Gallery.find({ user: req.user._id }).countDocuments(),
        images: await Image.find({ owner: req.user._id }).countDocuments()
      }
    });
  }
});

// POST - Usuwanie użytkownika przez admina
exports.user_delete_by_admin_post = asyncHandler(async (req, res, next) => {
  if (!req.user || req.user.username !== 'admin') {
    return res.status(403).send('Brak uprawnień');
  }

  const userId = req.params.id;
  
  // Admin nie może usunąć samego siebie
  if (userId === req.user._id.toString()) {
    return res.redirect('/users');
  }

  try {
    const userToDelete = await user.findById(userId);
    if (!userToDelete) {
      return res.redirect('/users');
    }

    // Usuń wszystkie zdjęcia użytkownika
    const userImages = await Image.find({ owner: userId });
    for (const img of userImages) {
      // Usuń plik fizyczny
      const fs = require('fs');
      const path = require('path');
      const filePath = path.join(__dirname, '../public/images', img.filename);
      if (fs.existsSync(filePath)) {
        fs.unlinkSync(filePath);
      }
    }
    await Image.deleteMany({ owner: userId });

    // Usuń wszystkie galerie użytkownika
    await Gallery.deleteMany({ user: userId });

    // Usuń wszystkie komentarze użytkownika
    await Comment.deleteMany({ author: userId });

    // Usuń konto użytkownika
    await user.findByIdAndDelete(userId);

    res.redirect('/users');
  } catch (error) {
    console.error('Błąd podczas usuwania użytkownika:', error);
    res.redirect('/users');
  }
});



