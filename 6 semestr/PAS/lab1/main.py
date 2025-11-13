#Oliwia Groszek, grupa czwartek 8:00
import socket
from PIL import Image
import ipaddress
import sys


def zad1():
    file_name = input("Podaj nazwę pliku: ")

    try:
        with open(file_name, 'r') as file:
            content = file.read()
        with open("lab1zad1.txt", 'w') as new_file:
            new_file.write(content)
        print("Zawartość pliku została zapisana")
    except FileNotFoundError:
        print("Nie ma pliku")

def zad2():
    filename = input("Podaj nazwę pliku graficznego: ")
    try:
        image = Image.open(filename)
        image.save("lab1zad2.png")
        print("Plik został zapisany")
    except FileNotFoundError:
        print("Nie ma takiego pliku")

def zad3():
    adress = input("Podaj adres IP: ")
    try:
        ip = ipaddress.ip_address(adress)
        print("Adres jest poprawny")
    except ValueError:
        print("Nie ma takiego adresu")

def zad4():
    ip = input("Podaj adres IP: ")
    try:
        hostname = socket.gethostbyaddr(ip)
        print(f"Hostname dla podanego IP to {hostname}")
    except socket.gaierror:
        print("Nie znaleziono hostname")

def zad5():
    hostname = input("Podaj hostname: ")
    try:
        ipaddress = socket.gethostbyname(hostname)
        print(f"IP dla podanego adresu to {ipaddress}")
    except socket.gaierror as e:
        print("Nie znaleziono IP")

def zad6():
    adress = sys.argv[1]
    port = int(sys.argv[2])
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((adress, port))
        print(f"Pomyślnie połączono z serwerem {adress} na porcie {port}")
    except socket.gaierror as e:
        print("Nie udało się połączyć")

def zad7():
    adress = sys.argv[1]
    try:
        ip = socket.gethostbyname(adress)
        for port in range(1, 65535):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.1)
            result = sock.connect_ex((ip, port))
            if result == 0:
                print(f"Port {port} otwarty")
            #else:
            #    print(f"Port {port}")
            sock.close()
    except socket.gaierror:
        print("Nieprawidłowy adres.")

#zad1()
#zad2()
#zad3()
#zad4()
#zad5()
#zad6()
zad7()