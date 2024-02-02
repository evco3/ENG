//
// Created by Evan Cohen on 2022-09-21.
//

#ifndef INC_320_WORKSPACE_INSULTGENERATOR_19ERC8_H
#define INC_320_WORKSPACE_INSULTGENERATOR_19ERC8_H

#endif //INC_320_WORKSPACE_INSULTGENERATOR_19ERC8_H

#include <iostream>
#include <string>
#include <vector>
#include <time.h>

using namespace std;

//creates variable for input file
const string insultFile("InsultsSource.txt");

//Exception class for file read error
class FileException: public exception {
public:
    char *what();
private:
};

//Exception class for number of insults error
class NumInsultsOutOfBounds: public exception{
public:
    char* what() ;
};

//Main Class for the insult generator
class InsultGenerator {
public:
    //Opens input file and creates vectors for each collumn of insult
    void initialize();

    //Used to randomly generate one insult
    string talkToMe();

    //Generates a vector of insult of a given length
    vector<string> generate(int);

    //Generates a vector of insults of a given lenght and saves them to an output file
    void generateAndSave(string, int);

private:
    //Generates random number for generating insults
    int randInt(int);

    //Vectors for each column of words
    vector<string> column1;
    vector<string> column2;
    vector<string> column3;
};