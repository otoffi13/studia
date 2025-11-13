package roguelike.resources;

import roguelike.logic.entities.Monster;
import roguelike.logic.level.Floor;
import roguelike.logic.level.Tile;

import java.util.Arrays;

public class Levels {


	public static final Floor BASE_LEVEL= new Floor(new String[] {


			"#####",
			"##^##",
			"#s..#",
			"#g..#",
			"#v..#",
			"##.##",
			"#...#",
			"#####"


		}, 2, 6);




	
	public static final Floor LEVEL_1= new Floor(new String[] {
			"###################",
			"########...########",
			"########...########",
			"####G..,.......####",
			"####,####.####.####",
			"####.#.......#.####",
			"#....#..###..#..p.#",
			"#..s....#^#......p#",
			"#....#..#.#..#....#",
			"####.#..,....#,####",
			"####.####.####.####",
			"####G..#...#..T.###",
			"#########.#########",
			"#########.#########",
			"###################"}, 9, 13,
			new Monster(Monster.Type.WOLF, 9, 3),
			new Monster(Monster.Type.WOLF, 10, 10));
	
	public static final Floor LEVEL_2 = new Floor(new String[] {
			"###################",
			"####^##############",
			"#s#...#.....,,....#",
			"#.#...#.#########.#",
			"#.##.##.#T.###.##.#",
			"#..........#..G...#",
			"#..###..#.##...##.#",
			"#..G....#.....#...#",
			"######,.###.###...#",
			"######...#....d...#",
			"######......d.....#",
			"###################"
		}, 9, 7,
			new Monster(Monster.Type.FOX, 11, 9),
			new Monster(Monster.Type.WOLF, 2, 8));

	public static final Floor LEVEL_3 = new Floor(new String[] {
			"###################",
			"#######..#######^##",
			"#p...........#gG..#",
			"#..#####.#.#.#....#",
			"#............####.#",
			"#.######.##.###^#.#",
			"#.T#.......###..#.#",
			"####.#.#.#####.##.#",
			"###......#...,.#..#",
			"####.###.......#.G#",
			"#.....p...#.,..#.T#",
			"###################"
		}, 1, 10,
			new Monster(Monster.Type.WOLF, 9, 6),
			new Monster(Monster.Type.FOX, 11, 2),
			new Monster(Monster.Type.RAT, 8, 5));
	
	public static final Floor LEVEL_4 = new Floor(new String[] {
			"###################",
			"#......####....v..#",
			"#.....s.......##..#",
			"#..#####.#.#.#....#",
			"#.....#...#.......#",
			"#......,,.#...##..#",
			"#.,.##....#...#...#",
			"#.......,.p..#....#",
			"###......#...,.#..#",
			"#..#...p.......#.G#",
			"#g...####..##....T#",
			"###################"
	}, 2, 10,
		new Monster(Monster.Type.WOLF, 11, 10),
		new Monster(Monster.Type.WOLF, 11, 3),
		new Monster(Monster.Type.RAT, 5, 10));
	
	public static final Floor LEVEL_5 = new Floor(new String[] {
			"#l##l#######l####",
			"#.........#T....#",
			"#.#######.####..#",
			"#......G#.......#",
			"#.##.#l##.#.#.#.#",
			"#.#.......#.#v#.#",
			"#.###.###....l#.#",
			"#........,#.....#",
			"###l#########^###"
	}, 1, 2, true,
		new Monster(Monster.Type.FOX, 4, 3),
		new Monster(Monster.Type.FOX, 15, 6));
	
	public static final Floor LEVEL_6 = new Floor(new String[] {
			"#################################",
			"#.^v##...,,....,....#####....g..#",
			"#.......###....##.,..G.....#....#",
			"#.,#......#..,...##....#####..###",
			"#..T##.##.,....#.....#.....#....#",
			"#################################"
		}, 29, 4,
			new Monster(Monster.Type.RAT, 9, 2),
			new Monster(Monster.Type.RAT, 2, 2),
			new Monster(Monster.Type.FOX, 4, 1),
			new Monster(Monster.Type.RAT, 23, 2));
	
	public static final Floor LEVEL_7 = new Floor(new String[] {
			"############l######",
			"####^###......#####",
			"#s#...#..l..,,....#",
			"#.#...#.###...l##.#",
			"#.#l.##.#T.###.##.#",
			"#..........#..G...#",
			"l..###..#.##...##.#",
			"#..G....#.....#...#",
			"###l##,.###.##l...#",
			"##.###...#...##...#",
			"##.###..##..d.....#",
			"##############l####"
	}, 4, 5, true,
		new Monster(Monster.Type.WOLF, 4, 3),
		new Monster(Monster.Type.FOX, 15, 6));

	public static final Floor LEVEL_8=new Floor(new String[]{
			"#########################",
			"#...G...#..,.p..#...s...#",
			"#.#####.........#...#...#",
			"#.^.......#....###......#",
			"#...##.........s....#...#",
			"#........##..,..........#",
			"######.###.....#....s...#",
			"#G.....#........###.....#",
			"#..T...#......#..v......#",
			"#########################"}, 2, 8,
			new Monster(Monster.Type.FOX, 13,2),
			new Monster(Monster.Type.FOX, 9, 1));

	public static final Floor LEVEL_9=new Floor(new String[]{
			"##########l##############l########",
			"#.............,.....G..#.........#",
			"#..##l####.,..........##...l..G..#",
			"#.G...#.....#...,..#.............#",
			"#...###...T#l.,....##.####l####,.#",
			"#...#...p..##.,....##...#.....#..#",
			"#...#.......#...,..#..........#.,l",
			"l........l###......#.T........#..#",
			"#.......s...........####..#...#,.#",
			"#.s...,........####.......#...^..l",
			"###########l###########l##########"}, 2, 1, true,
			new Monster(Monster.Type.FOX, 16,5),
			new Monster(Monster.Type.FOX, 12,9),
			new Monster(Monster.Type.RAT, 29, 2));

	public static final Floor LEVEL_10=new Floor(new String[]{
			"#####l#################",
			"#...#..##.G#.....#.^..#",
			"#..#ls....##....dl....l",
			"l......####l....###,.,#",
			"#....,.....#..........#",
			"#.........g....#...d..l",
			"####l########l#########"},2, 2, true,
			new Monster(Monster.Type.WOLF, 6, 2),
			new Monster(Monster.Type.FOX, 5, 6));


	public static final Floor LEVEL_11=new Floor(new String[]{
			"##################",
			"#.......#g.......#",
			"#....##.#....#..##",
			"#.......#...#...##",
			"#..#..G....#.,..##",
			"###...####.p...###",
			"##.........#######",
			"#T#..#v...#..,...#",
			"#.......,...#^.,.#",
			"##################"},1, 1,
			new Monster(Monster.Type.FOX, 2, 2),
			new Monster(Monster.Type.FOX, 4, 9),
			new Monster(Monster.Type.RAT, 13, 2));

	public static final Floor LEVEL_12=new Floor(new String[]{
			"################",
			"#.g.#^...####..#",
			"###.....,#...#.#",
			"#.....#....#.s.#",
			"##,.###....#.#.#",
			"#....#T....#..##",
			"#G.###...###v..#",
			"################"}, 7, 3,
			new Monster(Monster.Type.WOLF, 1, 1),
			new Monster(Monster.Type.WOLF, 7, 7),
			new Monster(Monster.Type.RAT, 14, 2));

}
