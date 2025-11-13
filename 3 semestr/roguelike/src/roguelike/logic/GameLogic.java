package roguelike.logic;

import java.io.*;
import java.util.Random;
import java.util.Scanner;

import javax.sound.sampled.*;
import javax.sound.sampled.spi.AudioFileReader;
import javax.swing.Timer;

import roguelike.gui.Renderer;
import roguelike.logic.entities.EntityTile;
import roguelike.logic.entities.Monster;
import roguelike.logic.entities.Player;
import roguelike.logic.items.Item;
import roguelike.logic.level.Floor;
import roguelike.logic.level.Tile;
import roguelike.logic.level.Tower;
import roguelike.logic.text.MessageBox;
import roguelike.resources.Items;
import roguelike.resources.Textures;

public class GameLogic {

	private static Timer timer;
	
	private static Random randomizer;
	
	private static Player player;
	private static Tower tower;
	private static Floor currentFloor;
	private static Monster[] activeMonsters;
	private static MessageBox messageBox;
	
	private static boolean onTitleScreen;

	public static void startGame() {
		Textures.init();
		
		init();
		
		timer = new Timer(20, new GameLoop());
		timer.start();
	}
	
	private static void init() {
		randomizer = new Random();

		tower = new Tower(randomizer);
		currentFloor = tower.getFloor(0);
		player = new Player("player", 2, 6);
		activeMonsters = currentFloor.getMonsters();
		messageBox = new MessageBox();
		
		onTitleScreen = true;
	}

	public static void genericLoop() {
		player.decreaseMotionOffset();
		for(Monster monster : activeMonsters) {
			monster.decreaseMotionOffset();
		}
	}

	public static void movePlayer(int dirX, int dirY) throws UnsupportedAudioFileException, LineUnavailableException, IOException {
		onTitleScreen = false;
		
		if(player.isInventoryOpen())
			return;
		
		if(checkIfPlayerDied())
			return;
		
		if(detectMonsterToFight(dirX, dirY)) {
			checkIfPlayerDied();
			return;
		}
		
		switch(getTileInFrontOfEntity(player, dirX, dirY).getName()) {
		case "floor":
			player.setPosition(player.getPosX()+dirX, player.getPosY()+dirY, true);
			break;
		case "wall":
		case "torch":
			messageBox.addMessage("Wszedłeś w drzewo!", 100);
			break;
		case "stairs":
			currentFloor = tower.getNextFloor();
			player.setPosition(currentFloor.getStartPosX(), currentFloor.getStartPosY(), false);
			activeMonsters = currentFloor.getMonsters();
			messageBox.addMessage("Wszedłeś do następnej części lasu", 200);
			player.addFloorCleared();
			//playSound("bush2.wav");
			break;
		case "trap":
			player.damage(randomizer.nextInt(2)+1);
			player.setPosition(player.getPosX()+dirX, player.getPosY()+dirY, true);
			currentFloor.disarmTrap(player.getPosX(), player.getPosY());
			messageBox.addMessage("Wszedłeś w ukrytą w trawie pułapkę na niedźwiedzie!", 200);
			break;
		}
		moveMonsters();
		
		checkIfPlayerDied();
	}

	public static void playSound(String name) throws LineUnavailableException, UnsupportedAudioFileException, IOException {
		Scanner scanner=new Scanner(System.in);
		File file = new File(".\\res\\sounds\\"+name);
		AudioInputStream audioInputStream = AudioSystem.getAudioInputStream(file);
		Clip clip = AudioSystem.getClip();
		clip.open(audioInputStream);
		clip.start();

	}

	private static Tile getTileInFrontOfEntity(EntityTile entity, int dirX, int dirY) {
		return currentFloor.getTileAt(entity.getPosX()+dirX, entity.getPosY()+dirY);
	}
	
	public static void openPlayerInventory() {
		if(player.getHealth() > 0)
			player.setInventoryOpen(!player.isInventoryOpen());
	}
	

	public static void handleInteration() {
		System.out.println("[Main][GameLogic]: Szukanie przedmiotu do podniesienia");
		pickupItem(player.getPosX()+1, player.getPosY());
		pickupItem(player.getPosX()-1, player.getPosY());
		pickupItem(player.getPosX(), player.getPosY()+1);
		pickupItem(player.getPosX(), player.getPosY()-1);
	}
	
	private static void pickupItem(int itemPosX, int itemPosY) {

		switch(currentFloor.getTileAt(itemPosX, itemPosY).getName()) {
		case "apple.g":
			if(player.giveItem(Items.HP_APPLE)) {
				currentFloor.removeCollectible(itemPosX, itemPosY);
				messageBox.addMessage("Podniosłeś jabłko!", 200);
			}
			else {
				messageBox.addMessage("Ekwipunek jest pełny!", 200);
			}
			break;
		case "gold_bag":
			int g = randomizer.nextInt(5)+3;
			player.giveGold(g);
			currentFloor.removeCollectible(itemPosX, itemPosY);
			messageBox.addMessage("Podniosłeś torbę zawierającą "+g+" złota!", 200);
			break;
		case "key":
			if(player.giveItem(Items.KEY)) {
				currentFloor.removeCollectible(itemPosX, itemPosY);
				messageBox.addMessage("Podniosłeś klucz", 200);
			}
			else {
				messageBox.addMessage("Ekwipunek jest pełny!", 200);
			}
			break;
		case "herbs_g":
			if(player.giveItem(Items.HERBS)) {
				currentFloor.removeCollectible(itemPosX, itemPosY);
				messageBox.addMessage("Podniosłeś zioła!", 200);
			}
			else {
				messageBox.addMessage("Ekwipunek jest pełny!", 200);
			}
			break;
		case "mint_g":
			if(player.giveItem(Items.MINT)) {
				currentFloor.removeCollectible(itemPosX, itemPosY);
				messageBox.addMessage("Podniosłeś miętę!", 200);
			}
			else {
				messageBox.addMessage("Ekwipunek jest pełny!", 200);
			}
			break;
		case "berries_g":
			if(player.giveItem(Items.BERRIES)) {
				currentFloor.removeCollectible(itemPosX, itemPosY);
				messageBox.addMessage("Podniosłeś jagody!", 200);
			}
			else {
				messageBox.addMessage("Ekwipunek jest pełny!", 200);
			}
			break;
		case "honey":
			if(player.giveItem(Items.HONEY)) {
				currentFloor.removeCollectible(itemPosX, itemPosY);
				messageBox.addMessage("Podniosłeś miód!", 200);
			}
			else {
				messageBox.addMessage("Ekwipunek jest pełny!", 200);
			}
			break;
		case "chest":
			switch(randomizer.nextInt(12)) {
			case 0:
				player.equipWeapon(Items.STONE);
				messageBox.addMessage("Znalazłeś kamień!", 200);
				break;
			case 1:
				player.equipArmor(Items.LIGHT_ARMOR);
				messageBox.addMessage("Znalazłeś ciepłą bluzę!", 200);
				break;
			case 2:
				player.equipWeapon(Items.STICK);
				messageBox.addMessage("Znalazłeś patyk!", 200);
				break;
			case 3:
				player.equipArmor(Items.BRONZE_ARMOR);
				messageBox.addMessage("Znalazłeś starą kurtkę!", 200);
				break;
			case 4:
				player.equipWeapon(Items.AXE);
				messageBox.addMessage("Znalazłeś siekierę!", 200);
				break;
			case 5:
				player.equipArmor(Items.MEDIEVAL_ARMOR);
				messageBox.addMessage("Znalazłeś lepszą kurtkę!", 200);
				break;
			case 6:
				player.equipWeapon(Items.SAW);
				messageBox.addMessage("Znalazłeś piłę!", 200);
				break;
			case 7:
				player.equipArmor(Items.MISTERIOUS_ARMOR);
				messageBox.addMessage("Znalazłeś kombinezon!", 200);
				break;
			case 8:
			case 9:
			case 10:
			case 11:
				int gold = randomizer.nextInt(9)+6;
				player.giveGold(gold);
				messageBox.addMessage("W skrzyni znalazłeś "+gold+" złota!", 200);
				break;
			}
			
			currentFloor.removeCollectible(itemPosX, itemPosY);
			break;
		}
	}
	

	public static void handleLeftClick(int mouseX, int mouseY) {
		System.out.println("[MAIN][GameLogic]: Trzymanie LPM");
		if(player.isInventoryOpen()) {
			if(Renderer.inventorySlot1.contains(mouseX, mouseY)) {
				usePlayerItem(0);
			}
			else if(Renderer.inventorySlot2.contains(mouseX, mouseY)) {
				usePlayerItem(1);
			}
			else if(Renderer.inventorySlot3.contains(mouseX, mouseY)) {
				usePlayerItem(2);
			}
		}
		
		if(player.getHealth() <= 0) {
			init();
		}
	}
	
	private static void usePlayerItem(int index) {
		Item item = player.getInventoryItem(index);
		
		if(item == null) return;
		
		if(item == Items.HP_APPLE) {
			player.heal(10);
			messageBox.addMessage("Zjadłeś jabłko!", 200);
		}
		else if(item == Items.KEY) {
			if(currentFloor.getTileAt(player.getPosX()+1, player.getPosY()).getName() == "locked_door") {
				currentFloor.openDoor(player.getPosX()+1, player.getPosY());
			}
			else if(currentFloor.getTileAt(player.getPosX()-1, player.getPosY()).getName() == "locked_door") {
				currentFloor.openDoor(player.getPosX()-1, player.getPosY());
			}
			else if(currentFloor.getTileAt(player.getPosX(), player.getPosY()+1).getName() == "locked_door") {
				currentFloor.openDoor(player.getPosX(), player.getPosY()+1);
			}
			else if(currentFloor.getTileAt(player.getPosX(), player.getPosY()-1).getName() == "locked_door") {
				currentFloor.openDoor(player.getPosX(), player.getPosY()-1);
			}
			else {
				messageBox.addMessage("Nie możesz tego tak użyć...", 200);
				return;
			}
			messageBox.addMessage("Użyłeś klucza do otwarcia drzwi!", 200);
		}
		else if(item == Items.HERBS) {
			player.addStrengthBuff();
			messageBox.addMessage("Zjadłeś zioła, siła została zwiększona!", 200);
		}
		else if(item == Items.BERRIES) {
			player.damage(5);
			messageBox.addMessage("Jedzenie nieznanych jagód nie było dobrym pomysłem, zatrułeś się!", 200);
		}
		else if(item == Items.MINT){
			player.addDefenceBuff();
			messageBox.addMessage("Zjadłeś miętę, twoja obrona wzrosła!", 200);
		}
		else if(item == Items.HONEY) {
			player.increaseHealth(5);
			messageBox.addMessage("Twoje zdrowie zostało zwiększone o 5 pkt!", 200);
		}
		player.removeItem(index);
	}
	
	private static void moveMonsters() throws UnsupportedAudioFileException, LineUnavailableException, IOException {
		for(Monster monster : activeMonsters) {
			if(monster.getHealth() <= 0)
				continue;
			
			if(!monster.shouldChasePlayer()) {
				switch(randomizer.nextInt(4)) {
				case 0:
					if(currentFloor.thereIsMonsterHere(monster.getPosX()+1, monster.getPosY())) {
						return;
					}
					else if(monster.getPosX()+1 == player.getPosX() && monster.getPosY() == player.getPosY()) {
						messageBox.addMessage("Zaatakowało Cię zwierzę!", 100);
						playSound("clap2.wav");
						player.damage(monster.getStrength() - player.getDefence()/3);
						if(player.damageArmor())
							messageBox.addMessage("Twoje ubranie zniszczyło się!", 200);
						break;
					}	
					if(getTileInFrontOfEntity(monster, 1, 0).getName() == "floor") {
						monster.setPosition(monster.getPosX()+1, monster.getPosY(), true);
						break;
					}
				case 1:
					if(currentFloor.thereIsMonsterHere(monster.getPosX()-1, monster.getPosY())) {
						return;
					}
					else if(monster.getPosX()-1 == player.getPosX() && monster.getPosY() == player.getPosY()) {
						messageBox.addMessage("Zaatakowało Cię zwierzę!", 100);
						playSound("clap2.wav");
						player.damage(monster.getStrength() - player.getDefence()/3);
						if(player.damageArmor())
							messageBox.addMessage("Twoje ubranie zniszczyło się!", 200);
						break;
					}	
					if(getTileInFrontOfEntity(monster, -1, 0).getName() == "floor") {
						monster.setPosition(monster.getPosX()-1, monster.getPosY(), true);
						break;
					}
				case 2:
					if(currentFloor.thereIsMonsterHere(monster.getPosX(), monster.getPosY()+1)) {
						return;
					}
					else if(monster.getPosX() == player.getPosX() && monster.getPosY()+1 == player.getPosY()) {
						messageBox.addMessage("Zaatakowało Cię zwierzę!", 100);
						playSound("clap2.wav");
						player.damage(monster.getStrength() - player.getDefence()/3);
						if(player.damageArmor())
							messageBox.addMessage("Twoje ubranie zniszczyło się!", 200);
						break;
					}	
					if(getTileInFrontOfEntity(monster, 0, 1).getName() == "floor") {
						monster.setPosition(monster.getPosX(), monster.getPosY()+1, true);
						break;
					}
				case 3:
					if(currentFloor.thereIsMonsterHere(monster.getPosX(), monster.getPosY()-1)) {
						return;
					}
					else if(monster.getPosX() == player.getPosX() && monster.getPosY()-1 == player.getPosY()) {
						messageBox.addMessage("Zaatakowało Cię zwierzę!", 100);
						playSound("clap2.wav");
						player.damage(monster.getStrength() - player.getDefence()/3);
						if(player.damageArmor())
							messageBox.addMessage("Twoje ubranie zniszczyło się!", 200);
						break;
					}	
					if(getTileInFrontOfEntity(monster, 0, -1).getName() == "floor") {
						monster.setPosition(monster.getPosX(), monster.getPosY()-1, true);
						break;
					}
				}
			} else {
				float angCoeff = -((float)player.getPosY()-(float)monster.getPosY())/((float)player.getPosX()-(float)monster.getPosX());
				
				if(angCoeff>-1 && angCoeff<1 && player.getPosX()>monster.getPosX()) {
					if(monster.getPosX()+1 == player.getPosX() && monster.getPosY() == player.getPosY()) {
						messageBox.addMessage("Zaatakowało Cię zwierzę!", 100);
						playSound("clap2.wav");
						player.damage(monster.getStrength() - player.getDefence()/3);
						if(player.damageArmor())
							messageBox.addMessage("Twoje ubranie zniszczyło się!", 200);
					}
					else if(getTileInFrontOfEntity(monster, 1, 0).getName() == "floor") {
						monster.setPosition(monster.getPosX()+1, monster.getPosY(), true);
					}
				}
				else if(angCoeff>-1 && angCoeff<1 && player.getPosX()<monster.getPosX()) {
					if(monster.getPosX()-1 == player.getPosX() && monster.getPosY() == player.getPosY()) {
						messageBox.addMessage("Zaatakowało Cię zwierzę!", 100);
						playSound("clap2.wav");
						player.damage(monster.getStrength() - player.getDefence()/3);
						if(player.damageArmor())
							messageBox.addMessage("Twoje ubranie zniszczyło się!", 200);
					}
					else if(getTileInFrontOfEntity(monster, -1, 0).getName() == "floor") {
						monster.setPosition(monster.getPosX()-1, monster.getPosY(), true);
					}
				}
				else if((angCoeff>1 || angCoeff<-1) && player.getPosY()>monster.getPosY()) {
					if(monster.getPosX() == player.getPosX() && monster.getPosY()+1 == player.getPosY()) {
						messageBox.addMessage("Zaatakowało Cię zwierzę!", 100);
						playSound("clap2.wav");
						player.damage(monster.getStrength() - player.getDefence()/3);
						if(player.damageArmor())
							messageBox.addMessage("Twoje ubranie zniszczyło się!", 200);
					}
					else if(getTileInFrontOfEntity(monster, 0, 1).getName() == "floor") {
						monster.setPosition(monster.getPosX(), monster.getPosY()+1, true);
					}
				}
				else if((angCoeff>1 || angCoeff<-1) && player.getPosY()<monster.getPosY()) {
					if(monster.getPosX() == player.getPosX() && monster.getPosY()-1 == player.getPosY()) {
						messageBox.addMessage("Zaatakowało Cię zwierzę!", 100);
						playSound("clap2.wav");
						player.damage(monster.getStrength() - player.getDefence()/3);
						if(player.damageArmor()) 
							messageBox.addMessage("Twoje ubranie zniszczyło się!", 200);
					}
					else if(getTileInFrontOfEntity(monster, 0, -1).getName() == "floor") {
						monster.setPosition(monster.getPosX(), monster.getPosY()-1, true);
					}
				}
			}
		}
	}
	

	private static boolean detectMonsterToFight(int dirX, int dirY) throws UnsupportedAudioFileException, LineUnavailableException, IOException {
		if(currentFloor.getMonsterAt(player.getPosX()+dirX, player.getPosY()+dirY) != null) {
			
			Monster fight = currentFloor.getMonsterAt(player.getPosX()+dirX, player.getPosY()+dirY);
			fight.damage(player.getStrength() - fight.getDefence()/3);
			
			if(player.damageWeapon())
				messageBox.addMessage("Twoja broń zniszczyła się", 200);
			
			if(fight.getHealth() <= 0) { //zwierze nie żyje
				currentFloor.killMonster(fight.getPosX(), fight.getPosY());
				int g = randomizer.nextInt(12)+8;
				player.giveGold(g);
				messageBox.addMessage("Zabiłeś zwierzę, które upuściło "+g+" złota!", 200);
			}
			else { //Monster is still alive after attack
				messageBox.addMessage("Zaatakowałeś zwierzę a ono ci oddało!", 200);
				playSound("clap2.wav");
				player.damage(fight.getStrength() - player.getDefence()/3);
				
				if(player.damageArmor())
					messageBox.addMessage("Twoje ubranie zniszczyło się!", 200);
			}
			
			if(dirX > 0) //Has attacked a monster on its left
				player.setMotionOffset(-16, 0);
			else if(dirX < 0) //Has attacked a monster on its right
				player.setMotionOffset(16, 0);
			else if(dirY > 0) //Has attacked a monster above
				player.setMotionOffset(0, -16);
			else if(dirY < 0) //Has attacked a monster below
				player.setMotionOffset(0, 16);
			
			return true;
		}
		return false;
	}
	
	private static boolean checkIfPlayerDied() {
		if(player.getHealth() <= 0) {
			messageBox.addMessage("Zginąłeś w lesie", 600);
			return true;
		}
		return false;
	}
	
	public static Player getPlayer() {
		return player;
	}
	
	public static Floor getCurrentFloor() {
		return currentFloor;
	}
	
	public static Monster[] getMonsters() {
		return activeMonsters;
	}
	
	public static MessageBox getMessageBox() {
		return messageBox;
	}
	
	public static boolean isOnTitleScreen() {
		return onTitleScreen;
	}
}
