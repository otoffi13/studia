package roguelike.logic.level;

import java.util.ArrayList;
import java.util.List;

import roguelike.logic.entities.Monster;

public class Floor {

	private Tile[][] floor;
	
	private int startPosX;
	private int startPosY;
	
	private List<Monster> monsters;
	
	private boolean dark;
	
	public Floor(String[] levelData, int startPosX, int startPosY, Monster... monsters) {
		this(levelData, startPosX, startPosY, false, monsters);
	}
	
	public Floor(String[] levelData, int startPosX, int startPosY, boolean isDark, Monster... monsters) {
		floor = new Tile[levelData.length][];
		
		for(int y=0;y<levelData.length;y++) {
			floor[y] = new Tile[levelData[y].length()];
			
			for(int x=0;x<levelData[y].length();x++) {
				switch(levelData[y].charAt(x)) {
				case '#':
					floor[y][x] = new Tile("wall", x, y);
					break;
				case '.':
					floor[y][x] = new Tile("floor", x, y);
					break;
				case '^':
					floor[y][x] = new Tile("stairs", x, y);
					break;
				case ',':
					floor[y][x] = new Tile("trap", x, y);
					break;
				case 'p':
					floor[y][x] = new Tile("apple.g", x, y);
					break;
				case 'G':
					floor[y][x] = new Tile("gold_bag", x, y);
					break;
				case 's':
					floor[y][x] = new Tile("herbs_g", x, y);
					break;
				case 'v':
					floor[y][x] = new Tile("berries_g", x, y);
					break;
				case 'd':
					floor[y][x] = new Tile("mint_g", x, y);
					break;
				case 'g':
					floor[y][x] = new Tile("honey", x, y);
					break;
				case 'e':
					floor[y][x] = new Tile("table2", x, y);
					break;
				case 'T':
					floor[y][x] = new Tile("chest", x, y);
					break;
				case 'l':
					floor[y][x] = new Tile("torch", x, y);
					break;
				}
			}
		}
		
		this.startPosX = startPosX;
		this.startPosY = startPosY;
		
		this.monsters = new ArrayList<Monster>();
		for(Monster one : monsters) {
			this.monsters.add(one);
		}
		
		this.dark = isDark;
	}
	
	public Floor(Floor copy) {
		floor = new Tile[copy.getSizeY()][];
		for(int y=0;y<copy.getSizeY();y++) {
			floor[y] = new Tile[copy.getSizeX()];
			for(int x=0;x<copy.getSizeX();x++) {
				floor[y][x] = new Tile(copy.getTileAt(x, y).getName(), x, y);
			}
		}
		
		this.startPosX = copy.getStartPosX();
		this.startPosY = copy.getStartPosY();
		
		this.monsters = copy.getMonstersList();
		
		this.dark = copy.isDark();
	}
	
	public int getSizeX() {
		return floor[0].length;
	}
	
	public int getSizeY() {
		return floor.length;
	}
	
	public Tile getTileAt(int x, int y) {
		return floor[y][x];
	}
	
	public int getStartPosX() {
		return startPosX;
	}
	
	public int getStartPosY() {
		return startPosY;
	}
	
	public Monster[] getMonsters() {
		Monster[] other = new Monster[monsters.size()];
		other = monsters.toArray(other);
		return other;
	}
	
	private List<Monster> getMonstersList(){
		return this.monsters;
	}
	
	public Monster getMonsterAt(int x, int y) {
		for(Monster monster : monsters) {
			if(monster == null)
				return null;
			
			if(monster.getPosX() == x && monster.getPosY() == y)
				return monster;
		}
		return null;
	}
	
	public boolean isDark() {
		return dark;
	}
	

	public boolean disarmTrap(int x, int y) {
		if(floor[y][x].getName() == "trap") {
			floor[y][x] = new Tile("floor", x, y);
			return true;
		}
		return false;
	}

	public boolean removeCollectible(int x, int y) {
		switch(floor[y][x].getName()) {
			case "apple.g":
			case "herbs_g":
			case "berries_g":
			case "gold_bag":
			case "mint_g":
			case "honey":
				floor[y][x] = new Tile("floor", x, y);
				return true;
		case "chest":
			floor[y][x] = new Tile("open_chest", x, y);
			return true;
		}
		return false;
	}

	public boolean openDoor(int x, int y) {
		if(floor[y][x].getName() == "locked_door") {
			floor[y][x] = new Tile("floor", x, y);
			return true;
		}
		return false;
	}

	public boolean killMonster(int x, int y) {
		for(int i=0;i<monsters.size();i++) {
			if(monsters.get(i).getPosX() == x && monsters.get(i).getPosY() == y) {
				monsters.remove(i);
				System.out.println("[GAMELOGIC][Floor]: Zwierzę zabite");
				return true;
			}
		}
		return false;
	}
	
	public boolean thereIsMonsterHere(int x, int y) {
		for(int i=0;i<monsters.size();i++) {
			if(monsters.get(i).getPosX() == x && monsters.get(i).getPosY() == y)
				return true;
		}
		return false;
	}
}