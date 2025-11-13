const Image = require("../models/image");
const Gallery = require("../models/gallery");
const Comment = require("../models/comment");
const Like = require("../models/like");
const asyncHandler = require("express-async-handler");
const { body, validationResult } = require("express-validator");
const multer = require('multer');
const path = require('path');

// Konfiguracja multer - tylko pliki jpg
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, path.join(__dirname, '../public/images'));
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + '-' + file.originalname);
  }
});
const fileFilter = (req, file, cb) => {
  if (file.mimetype === 'image/jpeg' || file.mimetype === 'image/jpg') {
    cb(null, true);
  } else {
    cb(new Error('Only JPG files are allowed!'), false);
  }
};
const upload = multer({ storage: storage, fileFilter: fileFilter });

// Eksportuj middleware upload
exports.upload = upload.single('file'); // 'file' = nazwa pola z formularza

// GET - formularz dodawania obrazka
exports.image_add_get = asyncHandler(async (req, res, next) => {
  let galleries;
  if (req.user.username === "admin") {
    galleries = await Gallery.find().populate("user").exec();
  } else {
    galleries = await Gallery.find({ user: req.user._id }).exec();
  }
  res.render("image_form", {
    title: "Dodaj zdjęcie",
    galleries,
    isAdmin: req.user.username === "admin"
  });
});

// POST - obsługa dodawania obrazka
exports.image_add_post = [
  exports.upload,
  body("name").trim().isLength({ min: 2 }).escape().withMessage("Nazwa zdjęcia jest za krótka."),
  body("description").trim().escape(),
  body("gallery").trim().isLength({ min: 1 }).escape().withMessage("Galerie musi być wybrana."),
  asyncHandler(async (req, res, next) => {
    if (!req.file) {
      let galleries;
      if (req.user.username === "admin") {
        galleries = await Gallery.find().populate("user").exec();
      } else {
        galleries = await Gallery.find({ user: req.user._id }).exec();
      }
      return res.render("image_form", {
        title: "Dodaj zdjęcie",
        messages: ["Plik jest wymagany."],
        galleries,
        isAdmin: req.user.username === "admin"
      });
    }

    const errors = validationResult(req);

    // Sprawdź, czy użytkownik ma prawo dodać obrazek do wybranej galerii
    const selectedGallery = await Gallery.findById(req.body.gallery).populate("user").exec();
    if (!selectedGallery) {
      let galleries;
      if (req.user.username === "admin") {
        galleries = await Gallery.find().populate("user").exec();
      } else {
        galleries = await Gallery.find({ user: req.user._id }).exec();
      }
      return res.render("image_form", {
        title: "Dodaj zdjęcie",
        messages: ["Wybrana galeria nie istnieje."],
        galleries,
        isAdmin: req.user.username === "admin"
      });
    }
    if (req.user.username !== "admin" && selectedGallery.user._id.toString() !== req.user._id.toString()) {
      let galleries;
      if (req.user.username === "admin") {
        galleries = await Gallery.find().populate("user").exec();
      } else {
        galleries = await Gallery.find({ user: req.user._id }).exec();
      }
      return res.render("image_form", {
        title: "Dodaj zdjęcie",
        messages: ["Możesz dodawać zdjęcia tylko do swoich galerii."],
        galleries,
        isAdmin: false
      });
    }

    // Tworzymy nowy dokument Image z nazwą pliku
    const newImage = new Image({
      name: req.body.name,
      description: req.body.description,
      filename: req.file.filename, // <-- tylko nazwa pliku!
      gallery: req.body.gallery,
      owner: req.user._id
    });

    if (!errors.isEmpty()) {
      let galleries;
      if (req.user.username === "admin") {
        galleries = await Gallery.find().populate("user").exec();
      } else {
        galleries = await Gallery.find({ user: req.user._id }).exec();
      }
      return res.render("image_form", {
        title: "Dodaj zdjęcie",
        image: newImage,
        galleries,
        messages: errors.array().map(e => e.msg),
        isAdmin: req.user.username === "admin"
      });
    }

    await newImage.save();
    res.render("image_form", {
      title: "Dodaj zdjęcie",
      messages: [`Zdjęcie "${newImage.name}" zostało dodane!`],
      galleries: req.user.username === "admin"
        ? await Gallery.find().populate("user").exec()
        : await Gallery.find({ user: req.user._id }).exec(),
      isAdmin: req.user.username === "admin"
    });
  })
];

// Lista wszystkich zdjęć
exports.image_list = async (req, res, next) => {
  try {
    const sortBy = req.query.sort || 'name'; // default sort by name
    const showMyOnly = req.query.my === 'true'; // filter to show only user's images
    const showLikedOnly = req.query.liked === 'true'; // filter to show only liked images
    
    let sortOptions = {};
    if (sortBy === 'date') {
      sortOptions = { date: -1 }; // newest first
    } else if (sortBy === 'likes') {
      // For likes sorting, we'll need to sort after getting like counts
      sortOptions = {}; // will be handled after fetching likes
    } else {
      sortOptions = { name: 1 }; // alphabetical
    }
    
    // Build query - filter by owner if requested
    let query = {};
    if (showMyOnly && req.user) {
      query.owner = req.user._id;
    }
    
    const images = await Image.find(query).populate('gallery').populate('owner').sort(sortOptions).exec();
    
    // Pobierz liczbę like'ów i status like'ów użytkownika dla każdego zdjęcia
    const imagesWithLikes = await Promise.all(images.map(async (image) => {
      const likeCount = await Like.countDocuments({ image: image._id });
      let userLiked = false;
      
      if (req.user) {
        const userLike = await Like.findOne({ 
          user: req.user._id, 
          image: image._id 
        });
        userLiked = !!userLike;
      }
      
      return {
        ...image.toObject(),
        likeCount: likeCount,
        userLiked: userLiked
      };
    }));
    
    // Filter by liked images if requested (mutually exclusive with my filter)
    let filteredImages = imagesWithLikes;
    if (showLikedOnly && req.user && !showMyOnly) {
      filteredImages = imagesWithLikes.filter(image => image.userLiked);
    }
    
    // Sort by likes if requested
    if (sortBy === 'likes') {
      filteredImages.sort((a, b) => b.likeCount - a.likeCount);
    }
    
    res.render('image_list', {
      images: filteredImages,
      currentUser: req.user,
      title: "Zdjęcia",
      currentSort: sortBy,
      showMyOnly: showMyOnly,
      showLikedOnly: showLikedOnly
    });
  } catch (err) {
    next(err);
  }
};

// Lista zdjęć w wybranej galerii
exports.images_in_gallery = asyncHandler(async (req, res, next) => {
  const galleryId = req.params.galleryId;
  const images = await Image.find({ gallery: galleryId }).populate('owner').exec();
  
  // Pobierz liczbę like'ów i status like'ów użytkownika dla każdego zdjęcia
  const imagesWithLikes = await Promise.all(images.map(async (image) => {
    const likeCount = await Like.countDocuments({ image: image._id });
    let userLiked = false;
    
    if (req.user) {
      const userLike = await Like.findOne({ 
        user: req.user._id, 
        image: image._id 
      });
      userLiked = !!userLike;
    }
    
    return {
      ...image.toObject(),
      likeCount: likeCount,
      userLiked: userLiked
    };
  }));
  
  res.render('image_list', {
    images: imagesWithLikes || [],
    currentUser: req.user,
    title: "Zdjęcia w galerii"
  });
});

// Usuwanie zdjęcia
exports.image_delete_post = async (req, res, next) => {
  try {
    const img = await Image.findById(req.params.id);
    if (!img) {
      return res.redirect('/images');
    }
    // Tylko admin lub właściciel zdjęcia może usunąć
    if (
      req.user.username !== 'admin' &&
      (!img.owner || img.owner.toString() !== req.user._id.toString())
    ) {
      return res.status(403).send('Brak uprawnień do usunięcia zdjęcia');
    }
    await Image.findByIdAndDelete(req.params.id);
    res.redirect('/images');
  } catch (err) {
    next(err);
  }
};

// GET - formularz edycji obrazka
exports.image_edit_get = asyncHandler(async (req, res, next) => {
  const image = await Image.findById(req.params.id).populate('gallery').populate('owner').exec();
  if (!image) {
    return res.redirect('/images');
  }
  
  // Tylko właściciel lub admin może edytować
  if (req.user.username !== 'admin' && !image.gallery.user._id.equals(req.user._id)) {
    return res.status(403).send('Brak uprawnień do edycji galerii');
  }

  let galleries;
  if (req.user.username === "admin") {
    galleries = await Gallery.find().populate("user").exec();
  } else {
    galleries = await Gallery.find({ user: req.user._id }).exec();
  }
  
  res.render("image_form", {
    title: "Edytuj zdjęcie",
    galleries,
    image: image,
    isAdmin: req.user.username === "admin"
  });
});

// POST - obsługa edycji obrazka
exports.image_edit_post = [
  body("name").trim().isLength({ min: 2 }).escape().withMessage("Nazwa zdjęcia jest za krótka."),
  asyncHandler(async (req, res, next) => {
    const errors = validationResult(req);
    
    const image = await Image.findById(req.params.id);
    if (!image) {
      return res.redirect('/images');
    }
    
    // Tylko właściciel lub admin może edytować
    if (req.user.username !== 'admin' && !image.gallery.user._id.equals(req.user._id)) {
      return res.status(403).send('Brak uprawnień do edycji galerii');
    }

    if (!errors.isEmpty()) {
      let galleries;
      if (req.user.username === "admin") {
        galleries = await Gallery.find().populate("user").exec();
      } else {
        galleries = await Gallery.find({ user: req.user._id }).exec();
      }
      return res.render("image_form", {
        title: "Edytuj zdjęcie",
        image: image,
        galleries,
        messages: errors.array().map(e => e.msg),
        isAdmin: req.user.username === "admin"
      });
    }

    // Zaktualizuj obrazek bezpośrednio (tylko nazwa i opis)
    await Image.findByIdAndUpdate(req.params.id, {
      name: req.body.name,
      description: req.body.description
      // Galeria pozostaje bez zmian
    }, {});
    
    // Pobierz zaktualizowany obrazek
    const updatedImage = await Image.findById(req.params.id).populate('gallery').populate('owner').exec();
    
    res.render("image_form", {
      title: "Edytuj zdjęcie",
      messages: [`Zdjęcie "${req.body.name}" zostało zaktualizowane!`],
      galleries: req.user.username === "admin"
        ? await Gallery.find().populate("user").exec()
        : await Gallery.find({ user: req.user._id }).exec(),
      image: updatedImage,
      isAdmin: req.user.username === "admin"
    });
  })
];

// GET - Wyświetlanie zdjęcia z komentarzami
exports.image_view = asyncHandler(async (req, res, next) => {
  const image = await Image.findById(req.params.id)
    .populate('gallery')
    .populate('owner')
    .exec();
  
  if (!image) {
    return res.redirect('/images');
  }

  // Pobierz komentarze dla tego zdjęcia
  const comments = await Comment.find({ image: req.params.id })
    .populate('author')
    .sort({ date: -1 })
    .exec();

  // Pobierz liczbę like'ów dla tego zdjęcia
  const likeCount = await Like.countDocuments({ image: req.params.id });

  // Sprawdź, czy aktualny użytkownik polubił to zdjęcie
  let userLiked = false;
  if (req.user) {
    const userLike = await Like.findOne({ 
      user: req.user._id, 
      image: req.params.id 
    });
    userLiked = !!userLike;
  }

  res.render('image_view', {
    title: image.name,
    image: image,
    comments: comments,
    currentUser: req.user,
    likeCount: likeCount,
    userLiked: userLiked
  });
});

// POST - Dodawanie komentarza
exports.comment_add_post = [
  body("content")
    .trim()
    .isLength({ min: 1, max: 500 })
    .escape()
    .withMessage("Komentarz musi mieć od 1 do 500 znaków."),
  
  asyncHandler(async (req, res, next) => {
    const errors = validationResult(req);
    
    const image = await Image.findById(req.params.imageId);
    if (!image) {
      return res.redirect('/images');
    }

    if (!errors.isEmpty()) {
      // Jeśli są błędy, przekieruj z powrotem do widoku zdjęcia
      const comments = await Comment.find({ image: req.params.imageId })
        .populate('author')
        .sort({ date: -1 })
        .exec();
      
      return res.render('image_view', {
        title: image.name,
        image: image,
        comments: comments,
        currentUser: req.user,
        messages: errors.array().map(e => e.msg)
      });
    }

    // Utwórz nowy komentarz
    const comment = new Comment({
      content: req.body.content,
      author: req.user._id,
      image: req.params.imageId,
      date: new Date()
    });

    await comment.save();
    
    // Przekieruj z powrotem do widoku zdjęcia
    res.redirect(`/images/view/${req.params.imageId}`);
  })
];

// POST - Usuwanie komentarza
exports.comment_delete_post = asyncHandler(async (req, res, next) => {
  const comment = await Comment.findById(req.params.commentId);
  
  if (!comment) {
    return res.redirect('/images');
  }

  // Tylko autor komentarza lub admin może go usunąć
  if (
    req.user.username !== 'admin' &&
    comment.author.toString() !== req.user._id.toString()
  ) {
    return res.status(403).send('Brak uprawnień do usunięcia komentarza');
  }

  await Comment.findByIdAndDelete(req.params.commentId);
  
  // Przekieruj z powrotem do widoku zdjęcia
  res.redirect(`/images/view/${comment.image}`);
});

// POST - Dodawanie/usuwanie like'a
exports.like_toggle_post = asyncHandler(async (req, res, next) => {
  if (!req.user) {
    return res.status(401).json({ error: 'Musisz być zalogowany, aby polubić zdjęcie' });
  }

  const imageId = req.params.imageId;
  const userId = req.user._id;

  // Sprawdź, czy zdjęcie istnieje
  const image = await Image.findById(imageId);
  if (!image) {
    return res.status(404).json({ error: 'Zdjęcie nie zostało znalezione' });
  }

  try {
    // Sprawdź, czy użytkownik już polubił to zdjęcie
    const existingLike = await Like.findOne({ user: userId, image: imageId });

    if (existingLike) {
      // Usuń like
      await Like.findByIdAndDelete(existingLike._id);
      const newLikeCount = await Like.countDocuments({ image: imageId });
      return res.json({ 
        liked: false, 
        likeCount: newLikeCount,
        message: 'Like usunięty'
      });
    } else {
      // Dodaj like
      const newLike = new Like({
        user: userId,
        image: imageId,
        date: new Date()
      });
      await newLike.save();
      const newLikeCount = await Like.countDocuments({ image: imageId });
      return res.json({ 
        liked: true, 
        likeCount: newLikeCount,
        message: 'Zdjęcie polubione'
      });
    }
  } catch (error) {
    console.error('Błąd podczas like\'owania:', error);
    return res.status(500).json({ error: 'Wystąpił błąd podczas like\'owania' });
  }
});

// GET - Pobieranie liczby like'ów (dla AJAX)
exports.get_like_count = asyncHandler(async (req, res, next) => {
  const imageId = req.params.imageId;
  
  const likeCount = await Like.countDocuments({ image: imageId });
  
  res.json({ likeCount: likeCount });
});