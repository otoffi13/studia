fn zad1(year:i16)->bool{
    if year%4==0 && year%100!=0 || year%400==0{
        println!("Leap");
        return true;
    }else{
        println!("Not leap");
        return false;
    }
}

fn zad2(){
    let year=2023;
    let month=3;
    if month==1 || month==3 || month==5 || month==7 || month==8 || month==10 || month==12 {
        println!("miesiac ma 31 dni!");
    } else if month==2 {
        let days;
        if zad1(year){
            days=29;
        } else { 
            days=28;
        }
        println!("Mieisac ma {} dni", days);
    }else{
    println!("Mieisac ma 30 dni");
    }
}


fn zad3(x: f64)->f64{
    (x*9.0)/5.0+32.0
}
fn zad3_nac(x: f64)->f64{
    (x-32.0)*5.0/9.0
}

fn zad5(h1: i32, m1:i32, s1:i32, h2: i32, m2:i32, s2:i32){
    let time1_s=h1*24*60+m1*60+s1;
    let time2_s=h2*24*60+m2*60+s2;
    let mut res=time2_s-time1_s;
    if res<0{
        res=-res;
    }
    let res_h = res / 3600;
    res -= res_h *3600;
    let res_m = res/60;
    res-=res_m*60;
    let res_s=res;
    
    println!("{}:{}:{}", res_h, res_m, res_s);
}

fn zad6( x: u128)->u128{
    if x==0 || x==1 {
        return 1;
    }
    x*zad6(x-1)
}

fn zad7(mut x: i32){
    while x>0 {
        println!("{}", x%10);
        x=x/10;
    }
}

fn zad8(mut n: u128)->u128{
   let mut sum=0;
   while n!=0 {
       sum += n% 10;
       n /= 10;
    }
    sum
}

fn zad9(c: i32){
    let a=1;
    let b=1;
    while a<c-1{
        while b<c-1{
            if a*a*b*b==c*C {
                println!("{} {} {}", a, b, c);
                b+=1;
            }
            a+=1;
        }
    }
}
fn main(){
}
