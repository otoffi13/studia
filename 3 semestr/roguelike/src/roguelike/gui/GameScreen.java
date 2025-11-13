package roguelike.gui;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Point;

import javax.swing.JPanel;

import roguelike.logic.GameLogic;

public class GameScreen extends JPanel {

	private Renderer renderer;
	

	public GameScreen() {
		super();
		this.setFocusable(true);
		this.addKeyListener(new Keyboard());
		this.addMouseListener(new Mouse());
		
		this.renderer = new Renderer();
		
		System.out.println("[GUI][GameScreen]: Otwarto GameScreen");
	}
	
	@Override
	protected void paintComponent(Graphics graphics) {
		super.paintComponent(graphics);
		
		try {
			
			graphics.setColor(new Color(38, 35, 34));
			graphics.fillRect(0, 0, Window.WIDTH, Window.HEIGHT);
	
			if(GameLogic.isOnTitleScreen()) {
				renderer.renderTitleScreen(graphics);
			} else {
				renderer.renderLevel(GameLogic.getCurrentFloor(), GameLogic.getPlayer(), graphics);
				renderer.renderPlayer(GameLogic.getPlayer(), graphics);
				renderer.renderMonsters(GameLogic.getMonsters(), GameLogic.getPlayer(), graphics);
				renderer.renderLight(GameLogic.getCurrentFloor(), GameLogic.getPlayer(), (Graphics2D) graphics);
				renderer.renderUI(GameLogic.getPlayer(), GameLogic.getCurrentFloor(), (Graphics2D) graphics, this.getMouseLocation());
				renderer.renderMessageBox(GameLogic.getMessageBox(), graphics);
			}
		} catch (Exception e) {
			System.err.println("\n[LOGIC][GameLoop]: Błąd systemu renderowania!\n");
			e.printStackTrace();
			System.exit(-1);
		}
		
		repaint();
	}
	
	public Point getMouseLocation() {
		if(getMousePosition() != null)
			return getMousePosition();
		else
			return new Point(-1, -1);
	}
}