//
// Created by Evan Cohen on 2022-09-21.
//


#include <istream>
#include <fstream>
#include <unordered_set>
#include <vector>
#include <string>
#include <cstdlib>
#include "insultgenerator_19erc8.h"


using namespace std;

char* FileException::what()  { return (char*)"There was an error reading this file"; }

char* NumInsultsOutOfBounds::what(){return (char*)"The number of insults is out of bounds";}


void InsultGenerator::initialize() {
    ifstream inputFile(insultFile);

    string word1;
    string word2;
    string word3;

    string insult;
    if(inputFile.fail())
        throw FileException();

    while(getline(inputFile, word1, '\t')){
        getline(inputFile, word2, '\t');
        getline(inputFile, word3);
        word3.pop_back();
        column1.push_back(word1);
        column2.push_back(word2);
        column3.push_back(word3);
    }

    inputFile.close();
}
int InsultGenerator::randInt(int val){
    return rand() % (val+1);
}


string InsultGenerator::talkToMe(){
    const int num1 = randInt(column1.size()-1);
    const int num2 = randInt(column2.size()-1);
    const int num3 = randInt(column3.size()-1);

    string insult ="Thou " + column1[num1] + " " + column2[num2] + " " + column3[num3] + "!";
    return insult;
}

vector<string> InsultGenerator::generate(int totalInsults){
    unordered_set<string> shakespeareInsults;

    if(totalInsults > 10000 || totalInsults < 1){
        throw NumInsultsOutOfBounds();
    }

    for(int i = 0; i< totalInsults; i++){
        string insult = talkToMe();

        if(shakespeareInsults.insert(insult).second == false){
            i--;
        }
    }
    vector<string> insultVector (shakespeareInsults.begin(), shakespeareInsults.end());
    sort(insultVector.begin(),insultVector.end());

    return insultVector;
}

void InsultGenerator::generateAndSave(string filename, int totalInsults){
    vector<string> listOfInsults(generate(totalInsults));

    ofstream outputFile(filename);

    if(outputFile.fail())
        throw FileException();

    int j = 0;
    while(j<totalInsults) {
        outputFile << listOfInsults[j] << endl;
        j++;
    }
    outputFile.close();
}
