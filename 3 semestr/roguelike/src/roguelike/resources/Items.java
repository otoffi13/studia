package roguelike.resources;

import roguelike.logic.items.Armor;
import roguelike.logic.items.Item;
import roguelike.logic.items.Weapon;

public class Items {

	public static final Item HP_APPLE = new Item("apple", "Zwykłe czerwone jabłko", "Odnawia 10 HP");
	public static final Item KEY = new Item("key", "Mały klucz", "Może zostać użyty aby otworzyć drzwi");
	public static final Item HERBS = new Item("herbs", "Zioła siły", "Zwiększa siłę");
	public static final Item BERRIES = new Item("berries", "Owoce", "Nieznany efekt");
	public static final Item MINT = new Item("mint", "Mięta", "Nieznany efekt");
	public static final Item HONEY = new Item("honey_g", "Miód pszczeli", "Zwiększa maksymalne zdrowie");
	
	public static final Weapon STONE = new Weapon("stone", "Mały kamień", 4, 10);
	public static final Weapon STICK = new Weapon("stick", "Patyk", 3, 7);
	public static final Weapon AXE = new Weapon("axe", "Siekiera", 7, 8);
	public static final Weapon SAW = new Weapon("saw", "Piła", 9, 12);
	//todo tekstury broni i zbroi
	public static final Armor LIGHT_ARMOR = new Armor("light_armor", "Ciepła bluza", 4, 10);
	public static final Armor BRONZE_ARMOR = new Armor("bronze_armor", "Stara kurtka", 5, 9);
	public static final Armor MEDIEVAL_ARMOR = new Armor("medieval_armor", "Lepsza kurtka", 7, 12);
	public static final Armor MISTERIOUS_ARMOR = new Armor("misterious_armor", "Kombinezon", 9, 15);
}
