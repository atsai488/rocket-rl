#include <Servo.h>

Servo leftArmOut;
Servo leftArmUp; 

Servo neckUpDown;
Servo neckRightLeft; 
 
Servo rightArmOut; 
Servo rightArmUp; 
bool running = false;
int motor = 1; 

void setup() {
  Serial.begin(9600);
  leftArmOut.attach(7);
  leftArmOut.write(100); 
  leftArmUp.attach(5); 
  leftArmUp.write(130); 
  neckRightLeft.attach(10); 
  neckRightLeft.write(90); 
  //neckUpDown.attach(2); //everytime i add this port for some reason the entire board doesnt work 
  //rightArmUp.attach(); //clicky servo, disconnected from arduino for now
  rightArmUp.write(170); 
  rightArmOut.attach(11); 
  rightArmOut.write(130); 
  Serial.println("Type 's' to start, 'x' to stop");
  Serial.println("Type 1-6, for which motor to run."); 
}

void loop() {
  if (Serial.available()) {
    char input = Serial.read();
    if (input == 's') { running = true; Serial.println("Started."); }
    if (input == 'x') { running = false; Serial.println("Stopped."); }
    if (input == '1') { motor = 1; Serial.println("Motor 1 selected."); }
    if (input == '2') { motor = 2; Serial.println("Motor 2 selected."); }
    if (input == '3') { motor = 3; Serial.println("Motor 3 selected."); }
    if (input == '4') { motor = 4; Serial.println("Motor 4 selected."); }
    if (input == '5') { motor = 5; Serial.println("Motor 5 selected."); }
    if (input == '6') { motor = 6; Serial.println("Motor 6 selected."); }
  }

  if (running && motor == 1) {
    leftArmOut.write(45); //for servo 3, assume by the side is starting pos 180, and 35 is max position
    delay(1000);
    leftArmOut.write(130);
    delay(1000);
    running = false;  // stop after one cycle
    Serial.println("Done. Type 's' to run again.");

  }

  if(running && motor == 2) {
    leftArmUp.write(45); 
    delay(1000);  
    leftArmUp.write(130);
    delay(1000);

    running = false;  // stop after one cycle
    Serial.println("Done. Type 's' to run again.");
    
    }

   if(running && motor == 3) {
    neckRightLeft.write(45); 
    delay(1000);  
    neckRightLeft.write(130);
    delay(1000);

    running = false;  // stop after one cycle
    Serial.println("Done. Type 's' to run again.");
    
    }

  if(running && motor == 4) {
    neckUpDown.write(45); 
    delay(1000); 
    neckUpDown.write(130); 
    delay(1000); 

    running = false;  // stop after one cycle
    Serial.println("Done. Type 's' to run again.");
    
    }

   if(running && motor == 5) {
    rightArmOut.write(45); 
    delay(1000); 
    rightArmOut.write(130); 
    delay(1000); 

    running = false;  // stop after one cycle
    Serial.println("Done. Type 's' to run again.");
    
    }

   if(running && motor == 6) {
    rightArmUp.write(45); 
    delay(1000); 
    rightArmUp.write(130); 
    delay(1000); 

    running = false;  // stop after one cycle
    Serial.println("Done. Type 's' to run again.");
    
    }
   
}
