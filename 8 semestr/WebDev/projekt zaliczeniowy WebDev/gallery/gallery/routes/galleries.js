const express = require('express');
const router = express.Router();
const gallery_controller = require("../controllers/galleryController");
const auth = require('../middleware/auth');

router.use(auth);

router.get('/', gallery_controller.gallery_list);
router.get('/gallery_add', gallery_controller.gallery_add_get);
router.post('/gallery_add', gallery_controller.gallery_add_post);
router.get('/:id/edit', gallery_controller.gallery_edit_get);
router.post('/:id/edit', gallery_controller.gallery_edit_post);
router.post('/:id/delete', gallery_controller.gallery_delete_post);

module.exports = router;

