package roguelike.gui;

import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;

public class Keyboard implements KeyListener {

	private static boolean[] keys;
	
	private static int delay;
	
	public Keyboard() {
		keys = new boolean[100];
		delay = 96;
	}
	
	@Override
	public void keyPressed(KeyEvent arg0) {
		keys[arg0.getKeyCode()] = true;
	}

	@Override
	public void keyReleased(KeyEvent arg0) {
		keys[arg0.getKeyCode()] = false;
	}
	

	public static boolean isKeyDown(int key) {
		if(keys[key] && delay <= 0) {
			delay = 96;
			return true;
		}
		else {
			delay--;
			return false;
		}
	}

	@Override
	public void keyTyped(KeyEvent arg0) {}
}
