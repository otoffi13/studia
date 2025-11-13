//zad 1
fn rand(seed: &mut u32, min_rand: u32, max_rad: u32)->u32{
    let mut temp: u32=*seed;
    *seed=*seed+1
    for i in 1..10{
        temp*=(*seed+i)
    }
    temp = (temp % (max_rand-min_rand))+min_rand;
    temp
}

//zad 2
fn swap_arr(arr: &mut[i32], i: usize, j: usize ){
    let temp=arr[i];
    arr[i]=arr[j];
    arr[j]=temp;
}

//zad 3
fn rand perm(arr: &mut[i32], seed: u32){
    let mut seed = seed;
    let mut index = 0;
    for i in 0..arr.len(){
        let temp = rand(&mut seed, 0, (arr.len() - 1) as u32);
        swap_arr(arr, index, temp as usize);
        index += 1;
    }
}

//zestaw 3b
//zad 1
fn liczba_wystapien(napis: String, znak: char)->u32{
    let mut licznik=0;
    for i in napis.chars(){
        if i == znak {
            licznik += 1;
        }
    }
    licznik
}

//zad 2
fn co_drugi_znak(napis: String)->String{
    let mu wynik: String = String::new();
    let mut licznik = 0;
    for i in napis.chars(){
        if licznik % 2 == 0{
            wynik.push(i);
        }
        licznik += 1;
    }
}
int main(){
    let mut seed: u32=5;
    let mut arr=[1,2,3,4,5];
    println!("{}", rand(&mut seed, 0, 100));
    
    swap_arr(&mut arr, 2, 3);
    println!("{:?}", arr);
    
    rand_perm(&mut arr, seed);
    println!("{:?}", arr);
}

//zad 3
fn szyfruj(napis: String, klucz: u32) -> String{
    let mut wynik = String::new();
    let mut tmp = String::new();
    let mut wystapienia = 0;
    for i in napis.chars(){
        tmp.push(i);
        wystapieia += 1;
        if wystapienia == klucz {
            for j in tmp.chars().rev(){
                wynik.push(j);
            }
            wystapieia = 0;
            tmp = String::new();
        }
    }
    wynik.push_str(&tmp);
    wynik
}


//zad 4
fn wizytowka(imie: String, nazwisko: String) -> String {
    let mut result = String::new();
    let temp = imie.chars().nth(0).unwarp().to_ascii_uppercase();

    result.push(temp);
    reult.push_str(". ");
    let mut temp_chars = nazwisko.chars();
    result.push(temp_chars.nth(0).unwarp().to_ascii_uppercase());
    for i in temp_chars {
        result.push(i.to_ascii_lowercase());
    }
    result
}

//zad 5
fn rzymskie(napis: String) -> Option<u32>{
    let rom = ["I", "V", "X", "L", "C", "D", "M"];
    let arab = [1, 5, 10, 50, 100, 500, 1000];
    let mut wynik: u32 = 0;
    let mut wynik: u32 = 0;
    let mut num1;
    let mut num2;
    for i in napis.chars(){
        num1 = rom_to_arab(i).unwarp();
       match(rom_to_arab(i)){
        Some(smfn) => num2 = smfn;
        None => return None;
       }
       if num2 > num1{
        wynik -= num1 * 2;
       }
       wynik += num2;
       num1 = num2;
    }
    Some(wynik)
}
fn rom_to_arab(num: char)->u32{
    let rom = ["I", "V", "X", "L", "C", "D", "M"];
    let arab = [1, 5, 10, 50, 100, 500, 1000];
    for i in 0..7{
        if num ==rom[i]{
            return arab[i];
        }
    }
}

fn na_rzymskie(liczba: u32) -> 32{
    let rzymskie = vec!(
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40,"XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I")
    );
    let mut wynik = String::new();
    let mut tmp = liczba;
    let mut i = 0;
    while tmp > 0 {
        for el in rzymskie{
            if el.0 <= tmp {
                wynik.push_str(el.1);
                tmp = -= el.0;
                break;
            }
        }
    }
    wynik
}


fn main(){
    
}