#kivy.require("1.10.0")
from kivy.app import App

from kivy.core.window import Window
#Window.clearcolor = (1, 1, 1, 1)

from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen, NoTransition
from kivy.uix.dropdown import DropDown
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput

sm = ScreenManager()

for i in range(6):
    screen = Screen(name='Title %d' % i)
    sm.add_widget(screen)

sm.current = 'Title 2'

class BootUp(Screen):
    pass

class SignUp(Screen):
    pass

class SignIn(Screen):
    pass

class MovieInput(Screen):
    pass

class MovieInput2(Screen):
    pass

class Waiting(Screen):
    pass

class CustomDropDown(DropDown):
    pass

dropdown = CustomDropDown()
mainbutton = Button(text='Choose a genre', size_hint=(None, None))
mainbutton.bind(on_release=dropdown.open)
dropdown.bind(on_select=lambda instance, x: setattr(mainbutton, 'text', x))

Builder.load_file("screens.kv")

sm = ScreenManager(transition=NoTransition())
sm.add_widget(BootUp(name='boot_up'))
sm.add_widget(SignUp(name='sign_up'))
sm.add_widget(SignIn(name='sign_in'))
sm.add_widget(MovieInput(name='movie_input'))
sm.add_widget(MovieInput2(name='movie_input2'))
sm.add_widget(Waiting(name='waiting'))

class MovieApp(App):
    def build(self):
        return sm

if __name__ == '__main__':
    MovieApp().run()

#return Image(source='Movie Placeholder Image.png')
#return Label(text='[color=000000]Hello[/color] [color=000000]World[/color]', markup = True)