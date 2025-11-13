const user = require("../models/user");
const Gallery = require("../models/gallery"); // <- poprawione!
const Image = require("../models/image");

const asyncHandler = require("express-async-handler");

exports.gallery_list = asyncHandler(async (req, res, next) => {
  const sortBy = req.query.sort || 'name'; // default sort by name
  const showMyOnly = req.query.my === 'true'; // filter to show only user's galleries
  
  let sortOptions = {};
  if (sortBy === 'date') {
    sortOptions = { date: -1 }; // newest first
  } else {
    sortOptions = { name: 1 }; // alphabetical
  }
  
  // Build query - filter by user if requested
  let query = {};
  if (showMyOnly && req.user) {
    query.user = req.user._id;
  }
  
  const all_galleries = await Gallery.find(query).populate("user").sort(sortOptions).exec();
  res.render("gallery_list", { 
    title: "Galerie", 
    gallery_list: all_galleries,
    currentUser: req.user,
    currentSort: sortBy,
    showMyOnly: showMyOnly
  });
});

// Import walidatora.
const { body, validationResult } = require("express-validator");

// GET - Kontroler wyświetlania formularza dodawania nowej galerii (metoda GET).
exports.gallery_add_get = asyncHandler(async (req, res, next) => {
  let all_users = [];
  if (req.user && req.user.username === "admin") {
    all_users = await user.find().sort({surname:1}).exec();
    // Admin widzi pełny formularz
    res.render("gallery_form", {
      title: "Dodaj galerię",
      users: all_users,
      gallery: {},
      isAdmin: true
    });
  } else {
    // Zwykły użytkownik widzi uproszczony formularz
    res.render("gallery_form_user", {
      title: "Dodaj galerię",
      user: req.user,
      gallery: {}
    });
  }
});

// POST - Kontroler (lista funkcji) obsługi danych z formularza dodawania nowej galerii (metoda POST).
exports.gallery_add_post = [
  // Ustaw właściciela galerii na aktualnie zalogowanego użytkownika
  asyncHandler(async (req, res, next) => {
    if (req.user) {
      req.body.g_user = req.user._id;
    }
    next();
  }),
  // Walidacja i sanityzacja danych z formularza.
  body("g_name", "Nazwa galerii jest za krótka.")
    .trim()
    .isLength({ min: 2 })
    .escape(),

  body("g_description")
    .trim()
    .escape(),

  // Nie sprawdzamy już wyboru użytkownika przez admina!
  // body("g_user", "Username must be selected.")
  //   .trim()
  //   .isLength({ min: 1 })
  //   .escape(),

  // Przetwarzanie po walidacji.
  asyncHandler(async (req, res, next) => {
    const errors = validationResult(req);

    // Ustaw właściciela na zalogowanego użytkownika
    const ownerId = req.user._id;
    const newgallery = new Gallery({
      name: req.body.g_name,
      description: req.body.g_description,
      user: ownerId,
      date: new Date(),
    });

    if (!errors.isEmpty()) {
      // Jeśli pojawiły się błędy - ponownie wyrenderuj formularz i wypełnij pola wprowadzonymi danymi po sanityzacji.
      let myMessages = [];
      errors.array().forEach(err => myMessages.push(err.msg));

      // Zwykły użytkownik widzi uproszczony formularz
      return res.render("gallery_form_user", {
        title: "Dodaj galerię:",
        gallery: newgallery,
        user: req.user,
        messages: myMessages,
      });
    }

    // Sprawdź, czy galeria o tej samej nazwie już istnieje dla tego użytkownika
    const galleryExists = await Gallery.findOne({
      name: req.body.g_name,
      user: ownerId,
    })
      .collation({ locale: "pl", strength: 2 })
      .exec();

    if (galleryExists) {
      return res.render("gallery_form_user", {
        title: "Dodaj galerię:",
        gallery: newgallery,
        user: req.user,
        messages: [`Galerie "${newgallery.name}" już istnieje!`]
      });
    }

    // Zapisz do bazy nową galerię.
    await newgallery.save();
    res.render("gallery_form_user", {
      title: "Dodaj galerię:",
      gallery: {},
      user: req.user,
      messages: [`Galerie "${newgallery.name}" została dodana!`],
    });
  }),
];

exports.gallery_delete_post = asyncHandler(async (req, res, next) => {
  const gallery = await Gallery.findById(req.params.id);
  if (!gallery) {
    return res.redirect('/galleries');
  }
  // Tylko właściciel lub admin może usunąć
  if (req.user.username !== 'admin' && !gallery.user.equals(req.user._id)) {
    return res.status(403).send('Brak uprawnień');
  }
  await Gallery.findByIdAndDelete(req.params.id);
  res.redirect('/galleries');
});

// GET - Kontroler wyświetlania formularza edycji galerii
exports.gallery_edit_get = asyncHandler(async (req, res, next) => {
  const gallery = await Gallery.findById(req.params.id).populate("user").exec();
  if (!gallery) {
    return res.redirect('/galleries');
  }
  
  // Tylko właściciel lub admin może edytować
  if (req.user.username !== 'admin' && !gallery.user._id.equals(req.user._id)) {
    return res.status(403).send('Brak uprawnień do edycji galerii');
  }

  let all_users = [];
  if (req.user && req.user.username === "admin") {
    all_users = await user.find().sort({surname:1}).exec();
    res.render("gallery_form", {
      title: "Edytuj galerię",
      users: all_users,
      gallery: gallery,
      isAdmin: true
    });
  } else {
    res.render("gallery_form_user", {
      title: "Edytuj galerię",
      user: req.user,
      gallery: gallery
    });
  }
});

// POST - Kontroler obsługi edycji galerii
exports.gallery_edit_post = [
  // Ustaw właściciela galerii na aktualnie zalogowanego użytkownika (dla zwykłych użytkowników)
  asyncHandler(async (req, res, next) => {
    if (req.user && req.user.username !== "admin") {
      req.body.g_user = req.user._id;
    }
    next();
  }),
  
  // Walidacja i sanityzacja danych z formularza
  body("g_name", "Nazwa galerii jest za krótka.")
    .trim()
    .isLength({ min: 2 })
    .escape(),

  body("g_description")
    .trim()
    .escape(),

  // Przetwarzanie po walidacji
  asyncHandler(async (req, res, next) => {
    const errors = validationResult(req);
    
    const gallery = await Gallery.findById(req.params.id);
    if (!gallery) {
      return res.redirect('/galleries');
    }
    
    // Tylko właściciel lub admin może edytować
    if (req.user.username !== 'admin' && !gallery.user.equals(req.user._id)) {
      return res.status(403).send('Brak uprawnień do edycji galerii');
    }

    const updatedGallery = new Gallery({
      name: req.body.g_name,
      description: req.body.g_description,
      user: req.user.username === "admin" ? req.body.g_user : req.user._id,
      date: gallery.date, // Zachowaj oryginalną datę
      _id: req.params.id // Zachowaj oryginalne ID
    });

    if (!errors.isEmpty()) {
      let myMessages = [];
      errors.array().forEach(err => myMessages.push(err.msg));

      if (req.user.username === "admin") {
        let all_users = await user.find().sort({surname:1}).exec();
        return res.render("gallery_form", {
          title: "Edytuj galerię:",
          users: all_users,
          gallery: updatedGallery,
          isAdmin: true,
          messages: myMessages,
        });
      } else {
        return res.render("gallery_form_user", {
          title: "Edytuj galerię:",
          gallery: updatedGallery,
          user: req.user,
          messages: myMessages,
        });
      }
    }

    // Sprawdź, czy galeria o tej samej nazwie już istnieje dla tego użytkownika (poza aktualną galerią)
    const galleryExists = await Gallery.findOne({
      name: req.body.g_name,
      user: req.user.username === "admin" ? req.body.g_user : req.user._id,
      _id: { $ne: req.params.id }
    })
      .collation({ locale: "pl", strength: 2 })
      .exec();

    if (galleryExists) {
      if (req.user.username === "admin") {
        let all_users = await user.find().sort({surname:1}).exec();
        return res.render("gallery_form", {
          title: "Edytuj galerię:",
          users: all_users,
          gallery: updatedGallery,
          isAdmin: true,
          messages: [`Gallery "${updatedGallery.name}" already exists!`]
        });
      } else {
        return res.render("gallery_form_user", {
          title: "Edytuj galerię:",
          gallery: updatedGallery,
          user: req.user,
          messages: [`Gallery "${updatedGallery.name}" already exists!`]
        });
      }
    }

    // Zaktualizuj galerię
    await Gallery.findByIdAndUpdate(req.params.id, {
      name: req.body.g_name,
      description: req.body.g_description,
      user: req.user.username === "admin" ? req.body.g_user : req.user._id
    });

    if (req.user.username === "admin") {
      let all_users = await user.find().sort({surname:1}).exec();
      res.render("gallery_form", {
        title: "Edytuj galerię:",
        users: all_users,
        gallery: {},
        isAdmin: true,
        messages: [`Gallery "${req.body.g_name}" updated!`],
      });
    } else {
      res.render("gallery_form_user", {
        title: "Edytuj galerię:",
        gallery: {},
        user: req.user,
        messages: [`Gallery "${req.body.g_name}" updated!`],
      });
    }
  }),
];