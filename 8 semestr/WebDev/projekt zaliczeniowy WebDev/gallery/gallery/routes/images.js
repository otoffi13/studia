const express = require('express');
const router = express.Router();
const gallery_controller = require("../controllers/galleryController");
const image_controller = require("../controllers/imageController");
const auth = require('../middleware/auth');

router.get("/", auth, image_controller.image_list);

// Dodawanie obrazka - tylko zalogowany użytkownik
router.get("/image_add", auth, image_controller.image_add_get);
router.post("/image_add", auth, image_controller.image_add_post);

// Edycja obrazka - tylko zalogowany użytkownik (musi być przed gallery/:galleryId)
router.get("/edit/:id", auth, image_controller.image_edit_get);
router.post("/edit/:id", auth, image_controller.image_edit_post);

// Usuwanie obrazka - tylko zalogowany użytkownik
router.post("/delete/:id", auth, image_controller.image_delete_post);

// Wyświetlanie zdjęcia z komentarzami
router.get("/view/:id", auth, image_controller.image_view);

// Komentarze - tylko zalogowany użytkownik
router.post("/:imageId/comment", auth, image_controller.comment_add_post);
router.post("/comment/:commentId/delete", auth, image_controller.comment_delete_post);

// Like system - tylko zalogowany użytkownik
router.post("/:imageId/like", auth, image_controller.like_toggle_post);
router.get("/:imageId/likes", image_controller.get_like_count);

// Przeglądanie obrazków w wybranej galerii (tylko zalogowany użytkownik)
router.get("/gallery/:galleryId", auth, image_controller.images_in_gallery);

module.exports = router;