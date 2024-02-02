//
// Created by Evan Cohen on 2022-10-18.
//

#ifndef INC_320_WORKSPACE_FRACTION_19ERC8_H
#define INC_320_WORKSPACE_FRACTION_19ERC8_H

#endif //INC_320_WORKSPACE_FRACTION_19ERC8_H

#include <string>
#include <iostream>

using namespace std;

//Exception thrown for improper fraction
class FractionException: public exception {
public:
    char *what();
private:
};

//Class of all instances of fractions
class Fraction{
public:
    //creates fraction of the numerator a
    Fraction(int a);

    //creates fraction of the numerator a and denominator b
    Fraction(int a, int b);

    //creates fraction 0/1
    Fraction();

    //numerator getter
    int numerator() const;

    //denominator getter
    int denominator() const;

    //Increments fraction buy given number
    Fraction& operator+=(const Fraction add);

    //increments fraction by 1
    Fraction& operator++();
    Fraction operator++(int unused);

    //compares fractions to find if they are larger, smaller or equal to each other
    int compareTo(const Fraction& f) const;

private:
    //numerator and denominator
    int top;
    int btm;

    //finds gcd to reduce fraction
    int gcd(int a, int b);

    //reduces fraction and proper sign placement
    void normalize();
};

//Mathematical operators
Fraction operator+ (const Fraction& a,const Fraction& b);
Fraction operator- (const Fraction& a,const Fraction& b);
Fraction operator- (const Fraction& a);
Fraction operator* (const Fraction& a,const Fraction& b);
Fraction operator/ (const Fraction& a,const Fraction& b);

//input and output operator
ostream& operator<< (ostream& out,const Fraction& f);
istream& operator>> (istream& in, Fraction& f);

//Comparison operators
bool operator <  (const Fraction& a,const Fraction& b);
bool operator <= (const Fraction& a,const Fraction& b);
bool operator >  (const Fraction& a,const Fraction& b);
bool operator >= (const Fraction& a,const Fraction& b);
bool operator == (const Fraction& a,const Fraction& b);
bool operator != (const Fraction& a,const Fraction& b);
