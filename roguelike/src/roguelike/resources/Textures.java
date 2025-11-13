package roguelike.resources;

import java.awt.image.BufferedImage;

import java.io.File;
import java.io.IOException;

import java.util.HashMap;

import javax.imageio.ImageIO;

public class Textures {

	private static HashMap<String, BufferedImage> sprites;

	public static void init() {
		sprites = new HashMap<String, BufferedImage>();
		
		File folder = new File(".\\res\\textures");
		
		for(File file : folder.listFiles()) {
			try {
				sprites.put(file.getName().replaceAll(".png", ""), ImageIO.read(file));
			} catch (IOException e) {
				System.err.println("[UTILS][Textures]: Błąd w odczycie "+file.getName());
			}
		}
		
		System.out.println("[UTILS][Textures]: Zakończono czytanie plików");
	}

	public static BufferedImage getSprite(String name) {
		BufferedImage sprite = sprites.get(name);
		if(sprite != null) return sprite;
		else return sprites.get("error");
	}
}