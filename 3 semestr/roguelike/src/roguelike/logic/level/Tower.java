package roguelike.logic.level;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import roguelike.resources.Levels;

public class Tower {

	private List<Floor> floors;
	private int floorAt;
	

	public Tower(Random randomizer) {
		this.floors = new ArrayList<Floor>();
		List<Floor> temporaryTower = new ArrayList<>();
		this.floorAt = 0;
		
		this.floors.add(Levels.BASE_LEVEL);
		temporaryTower.add(Levels.LEVEL_1);
		temporaryTower.add(Levels.LEVEL_2);
		temporaryTower.add(Levels.LEVEL_3);
		temporaryTower.add(Levels.LEVEL_4);
		temporaryTower.add(Levels.LEVEL_5);
		temporaryTower.add(Levels.LEVEL_6);
		temporaryTower.add(Levels.LEVEL_7);
		temporaryTower.add(Levels.LEVEL_8);
		temporaryTower.add(Levels.LEVEL_9);
		temporaryTower.add(Levels.LEVEL_10);
		temporaryTower.add(Levels.LEVEL_11);
		temporaryTower.add(Levels.LEVEL_12);
		
		while(!temporaryTower.isEmpty()) {
			int choice = temporaryTower.size() == 1 ? 0 : randomizer.nextInt(temporaryTower.size()-1)+1;
			this.floors.add(temporaryTower.get(choice));
			temporaryTower.remove(choice);
		}
	}

	public Floor getFloor(int index) {
		return floors.get(index);
	}
	

	public int getTowerHeight() {
		return floors.size();
	}

	public int getFloorAt() {
		return floorAt;
	}

	public Floor getNextFloor() {
		floorAt++;
		
		if(floorAt == floors.size())
			floorAt--;
		
		return floors.get(floorAt);
	}
	

	public Floor getPreviousFloor() {
		if(floorAt != 0)
			floorAt--;
		
		return floors.get(floorAt);
	}
}