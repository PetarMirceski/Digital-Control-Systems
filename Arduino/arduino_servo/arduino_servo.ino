#include <Servo.h>
Servo myservo;  // Define a sevo object
String inByte; // Define a string that reads from the Serial Port 
int pos;  // Define the position of the servo

void setup() {

  myservo.attach(9);  // The servo is attached to pin number 9
  Serial.begin(9600); // Initialising serial communucation 
  myservo.write(30);  // Writing the initial beam offset 
}

void loop()
{
  if (Serial.available()) // if data available in serial port
  {
    inByte = Serial.readStringUntil('\n'); // read data until newline
    pos = inByte.toInt();   // change datatype from string to integer
    myservo.write(pos);     // move servo
  }
}
