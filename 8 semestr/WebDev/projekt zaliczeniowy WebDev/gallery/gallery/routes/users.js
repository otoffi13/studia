const express = require('express');
const router = express.Router();
const user_controller = require("../controllers/userController");
const auth = require('../middleware/auth');

// Dostępne dla niezalogowanych
router.get('/login', user_controller.user_login_get);
router.post('/login', user_controller.user_login_post);
router.get('/user_add', user_controller.user_add_get);
router.post('/user_add', user_controller.user_add_post);
router.get('/logout', user_controller.user_logout_get);

// Wszystko poniżej wymaga zalogowania
router.use(auth);

router.get('/', user_controller.user_list);

// Zarządzanie kontem użytkownika
router.get('/profile', user_controller.user_profile_get);
router.post('/profile', user_controller.user_profile_post);
router.get('/delete-account', user_controller.user_delete_account_get);
router.post('/delete-account', user_controller.user_delete_account_post);

// Admin routes
router.post('/delete/:id', user_controller.user_delete_by_admin_post);

module.exports = router;