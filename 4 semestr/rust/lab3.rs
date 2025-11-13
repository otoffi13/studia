fn g(x: f64) -> f64{
    x=x+5;
    x  //<-to jest zwrócenie wartości xd
}

fn f_rosnaca(x: f64, epsilon: f64, f: fn(f64)->f64) -> bool{
    let epsilon=0.0000001;
    let wynik=f(x);
    let wynik2=f(x+epsilon)
    if(wynik > wynik2){
        return false;
    }else{
         return true;
    } 
}

fn met_newt(x0: f64, epsilon: f64, N: u128, f: fn(f64)->f64)->f64{
    let mut curr_point=x0;
    let mut delta: f64=1.0;
    let mut counter: u128=0;
    let mut prev_right: bool=true
    loop{
        let mut rising=f_jest_rosnaca(curr_point, epsilon);
        if counter==N{
            curr_point
        }
        if rising{
            if f(curr_point)>0{
                curr_point-=delta;
            }else{
                curr_point+=delta;
            }
        }else{
            if f(curr_point)>0{
                curr_point+=delta;
            }else{
                curr_point-=delta;
            }
    }   
    counter+=1;
    }
}

fn main(){
}