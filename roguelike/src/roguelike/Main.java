package roguelike;

import roguelike.gui.Window;
import roguelike.logic.GameLogic;

public class Main {

	public static void main(String[] args) {
		try {
			
			System.out.println("[MAIN]:Uruchamianie...");
			
			Window.create();
			GameLogic.startGame();
			Window.setVisible();
			
			System.out.println("[MAIN]: Uruchomiono!");
			
		} catch(Exception e) {
			System.err.println("\n[MAIN]: Błąd podczas inicjalizacji\n");
			e.printStackTrace();
			System.exit(-1);
		}
	}
}