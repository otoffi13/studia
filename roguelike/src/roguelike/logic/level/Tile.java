package roguelike.logic.level;

import java.sql.Array;

public class Tile {

	private String name;
	
	protected int worldPosX;
	protected int worldPosY;
	
	private boolean collectible;
	

	public Tile(String name, int posX, int posY) {
		this.name = name;
		this.worldPosX = posX;
		this.worldPosY = posY;
		
		if(name == "apple_g" || name == "gold_bag_g" || name == "key_g" || name == "herbs_g" || name == "berries_g" || name == "mint_g" || name == "honey_g" || name == "chest")
			this.collectible = true;
	}
	
	public String getName() {
		return name;
	}
	
	public int getPosX() {
		return worldPosX;
	}
	
	public int getPosY() {
		return worldPosY;
	}
	
	public boolean isCollectible() {
		return collectible;
	}

}
