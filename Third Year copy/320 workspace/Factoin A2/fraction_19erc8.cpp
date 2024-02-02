 //
// Created by Evan Cohen on 2022-10-18.
//

#include "fraction_19erc8.h"


using namespace std;


char* FractionException::what()  {return (char*)"The denominator cannot be 0."; }


Fraction::Fraction(int a){
    top = a;
    btm = 1;
}

Fraction::Fraction(int a, int b){
    top = a;
    btm = b;

    if(btm == 0)
        throw FractionException();
    normalize();
}

Fraction::Fraction(){
    top = 0;
    btm = 1;
}

int Fraction::numerator() const{
    return top;
};

int Fraction::denominator() const{
    return btm;
};

int Fraction::gcd(int n, int m){
    if(m <= n && n%m == 0)
        {return m;}
    else if(m>n)
        return gcd(m,n);
    else
        return gcd(m, n%m);
};

void Fraction::normalize(){
    int sign = 1;
    if(top < 1){
        top = -top;
        sign =-1;
    }
    if(btm < 1){
        btm = -btm;
        sign = -sign;
    }

    int commonDivisor = 1;
    if(top>0)
        commonDivisor = gcd(top,btm);

    top = top/commonDivisor * sign;
    btm = btm/commonDivisor;
};

Fraction& Fraction::operator++(){
    top += btm;
    return *this;
}

Fraction Fraction::operator++(int unused) {
    Fraction copy(top,btm);
    top += btm;
    return copy;
}

Fraction& Fraction::operator+=(const Fraction a) {
    top = top * a.denominator() + btm * a.numerator();
    btm *= a.denominator();
    normalize();
    return *this;
}

int Fraction::compareTo(const Fraction& f) const {
    int result =   (top * f.denominator() - btm * f.numerator());
    return result;
}

Fraction operator+(const Fraction& a,const Fraction& b){
    int num = a.numerator() * b.denominator() + a.denominator() * b.numerator();
    int den = a.denominator() * b.denominator();
    Fraction sum(num,den);

    return sum;
}

Fraction operator-(const Fraction& a,const Fraction& b){
    int num = a.numerator() * b.denominator() - a.denominator() * b.numerator();
    int den = a.denominator() * b.denominator();
    Fraction difference(num,den);

    return difference;
}

Fraction operator-(const Fraction& a){
    Fraction difference(-a.numerator(),a.denominator());

    return difference;
}

Fraction operator*(const Fraction& a,const Fraction& b){
    int num = a.numerator() * b.numerator();
    int den = a.denominator() * b.denominator();
    Fraction mult(num,den);

    return mult;
}

Fraction operator/(const Fraction& a,const Fraction& b){
    int num = a.numerator() * b.denominator();
    int den = a.denominator() * b.numerator();
    Fraction div(num,den);

    return div;
}

bool operator < (const Fraction& a,const Fraction& b){
    return a.compareTo(b) < 0;
}
bool operator <= (const Fraction& a,const Fraction& b){
    return a.compareTo(b) <= 0;
}

bool operator > (const Fraction& a,const Fraction& b){
    return a.compareTo(b) > 0;
}

bool operator >= (const Fraction& a,const Fraction& b){
    return a.compareTo(b) > 0;
}

bool operator == (const Fraction& a,const Fraction& b){
    return a.compareTo(b) == 0;
}

bool operator != (const Fraction& a,const Fraction& b){
    return a.compareTo(b) != 0;
}

ostream& operator << (ostream& out,const Fraction& f){
    out << f.numerator() << "/" << f.denominator();
    return out;
}

istream& operator >> (istream& in, Fraction& f){
    string fractionIn;
    in >> fractionIn;

    size_t numState = fractionIn.find('/');
    if(numState == string::npos){
        try{
            int num = stoi(fractionIn);
            f = Fraction(num);
            return in;
        }catch (exception& e){
            throw FractionException();
        }
    }else {
        if(numState + 1 >= fractionIn.length()){throw FractionException();}

        try{
            int top = stoi(fractionIn.substr(0,numState));
            int btm = stoi(fractionIn.substr(numState + 1, fractionIn.size() - numState - 1));
            f = Fraction(top,btm);
            return in;
        } catch (exception& e){
            (throw FractionException());
        }
    }
}



