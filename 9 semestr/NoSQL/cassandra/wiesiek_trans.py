from cassandra.cluster import Cluster
from uuid import UUID

cluster = Cluster(['127.0.0.1'], port=9042)
session = cluster.connect('wiesiektrans')

def show_drivers():
    print("\n--- Lista kierowców ---")
    rows = session.execute("SELECT * FROM drivers;")
    for row in rows:
        print(row)

def filter_drivers_by_skill():
    skill = input("Podaj umiejętność (np. lokomotywa_elektryczna): ")
    print(f"\n--- Kierowcy z umiejętnością: {skill} ---")

    query = """
    SELECT * FROM drivers
    WHERE skills CONTAINS %s
    ALLOW FILTERING;
    """
    rows = session.execute(query, [skill])
    for row in rows:
        print(row)

def show_vehicles():
    print("\n--- Lista pojazdów ---")
    rows = session.execute("SELECT * FROM vehicles;")
    for row in rows:
        print(row)


def show_current_vehicle_position():
    vehicle_id = input("Podaj UUID pojazdu: ")
    try:
        vehicle_uuid = UUID(vehicle_id)
    except ValueError:
        print("Niepoprawny UUID")
        return

    query = """
    SELECT * FROM vehicle_current_position
    WHERE vehicle_id = %s;
    """
    row = session.execute(query, [vehicle_uuid]).one()

    if row:
        print("\n--- Aktualna pozycja pojazdu ---")
        print(row)
    else:
        print("Brak danych dla tego pojazdu")


def show_vehicle_history():
    vehicle_id = input("Podaj UUID pojazdu: ")
    date = input("Podaj datę (YYYY-MM-DD): ")

    try:
        vehicle_uuid = UUID(vehicle_id)
    except ValueError:
        print("Niepoprawny UUID")
        return

    query = """
    SELECT * FROM vehicle_position_history
    WHERE vehicle_id = %s AND date = %s;
    """
    rows = session.execute(query, [vehicle_uuid, date])

    print(f"\n--- Historia pojazdu {vehicle_id} z dnia {date} ---")
    for row in rows:
        print(row)


def show_driver_availability():
    driver = input("Podaj nazwę kierowcy: ")
    rows = session.execute(
        "SELECT * FROM driver_availability WHERE driver_name = %s;",
        [driver]
    )
    for row in rows:
        print(row)


def show_vehicle_availability():
    vehicle_id = input("Podaj UUID pojazdu: ")
    try:
        vehicle_uuid = UUID(vehicle_id)
    except ValueError:
        print("Niepoprawny UUID")
        return

    rows = session.execute(
        "SELECT * FROM vehicle_availability WHERE vehicle_id = %s;",
        [vehicle_uuid]
    )
    for row in rows:
        print(row)

def menu():
    print("""
=== WiesiekTrans – aplikacja administratora ===
1. Wyświetl wszystkich kierowców
2. Filtruj kierowców po umiejętności
3. Wyświetl pojazdy
4. Pokaż aktualną pozycję pojazdu
5. Pokaż historię pojazdu z danego dnia
6. Sprawdź dostępność kierowcy
7. Sprawdź dostępność pojazdu
0. Wyjście
""")

while True:
    menu()
    choice = input("Wybierz opcję: ")

    if choice == '1':
        show_drivers()
    elif choice == '2':
        filter_drivers_by_skill()
    elif choice == '3':
        show_vehicles()
    elif choice == '4':
        show_current_vehicle_position()
    elif choice == '5':
        show_vehicle_history()
    elif choice == '6':
        show_driver_availability()
    elif choice == '7':
        show_vehicle_availability()
    elif choice == '0':
        print("Zamykanie aplikacji")
        break
    else:
        print("Niepoprawna opcja")
