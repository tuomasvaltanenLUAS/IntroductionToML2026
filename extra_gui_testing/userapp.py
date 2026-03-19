# PART 1: IMPORTS AND LOAD THE MODEL FROM FILE
import tkinter
import pandas as pd
from joblib import load
model = load("linearreg1.joblib")

print("Should work?")

# how to run this:
# in your terminal, make sure venv is selected
# then select the folder where your userapp.py is 
# e.g. in this case: cd extra_gui_testing
# after this:
# python userapp.py

# PART 2: TEST THE MODEL FIRST WITHOUT PROCEEDING FURTHER (tester_row)
# usually in a GUI application
# we save the earlier model-object into a file (using joblib-module)
# and in a separate GUI-application => we load up the saved model from the file
# and use the model just like here below:

# this variable could be connected
# to a user interface (textbox etc.)
test_experience = 9

# map all the variables from the user
# into a Python dictionary
# the variable names have to match with the original dataset
tester_row = {
    'YearsExperience': test_experience
}

# convert to pandas-format
tester_row = pd.DataFrame([tester_row])

# get the output/result/answer from the model
# based on the user's new data (from above code cell)
result = model.predict(tester_row)[0]

print()
print(f"Predicted salary with {test_experience} years of work experience:")
print(f"{round(float(result), 2)} $")
print("----------------")

# the model seems to work as expected!!

# PART 3: CREATE THE GUI
# let's use this for starters:
# https://www.geeksforgeeks.org/how-to-set-text-of-tkinter-text-widget-with-a-button/

# Creating the GUI window.
window = tkinter.Tk()
window.title("Yearly salary linear regression model GUI application")
window.geometry("800x600")
window.option_add("*font", "lucida 20 bold")
 
# the label is the visual component shown in screen
title = tkinter.Label(window, text="Work experience")
title.pack(pady=10)

# Creating our text widget.
entry_experience = tkinter.Entry(window)
entry_experience.pack(pady=0)

# result_var is the variable we change
# when the model gives output
result_var = tkinter.StringVar()

# the label is the visual component shown in screen
label = tkinter.Label(window, textvariable=result_var)

# let's set up some default value for user
result_var.set("Waiting for user input...")

# set to GUI
label.pack(pady=20)

 
# Creating the function to set the text 
# with the help of button
def set_text_by_button():
 
    # inform the user if they provided wrong kind of data
    if not entry_experience.get().isnumeric():
        result_var.set("Incorrect value, use numbers.")
        entry_experience.configure(foreground="red")
    else:
        entry_experience.configure(foreground="black")

    test_experience = int(entry_experience.get())

    # map all the variables from the user
    # into a Python dictionary
    # the variable names have to match with the original dataset
    tester_row = {
        'YearsExperience': test_experience
    }

    # convert to pandas-format
    tester_row = pd.DataFrame([tester_row])

    # get the output/result/answer from the model
    # based on the user's new data (from above code cell)
    result = model.predict(tester_row)[0]

    result_text = f"{round(float(result), 2)} $"
    # Delete is going to erase anything
    # in the range of 0 and end of file,
    # The respective range given here
    entry_experience.delete(0, "end")
     
    # Insert method inserts the text at
    # specified position, Here it is the
    # beginning
    result_var.set(result_text)
 
# Setting up the button, set_text_by_button() 
# is passed as a command
set_up_button = tkinter.Button(window, height=1, width=16, text="Get my salary!", 
                    command=set_text_by_button)
 
set_up_button.pack(pady=20)
 
# if Enter is pressed => launch the model function
def handle_enter(event):
    set_text_by_button()

# register in the application window
# => if return (enter) is pressed => run a function
window.bind('<Return>', handle_enter)

window.mainloop()