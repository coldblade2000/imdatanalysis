#kivy.require("1.10.0")
from kivy.app import App
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import ScreenManager, Screen, NoTransition

Window.clearcolor = (0.176, 0.203, 0.211, 1)

sm = ScreenManager()

for i in range(10):
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


class MovieInput2(Screen):  # Actual list
    pass

class Waiting(Screen):
    pass

class HowTo(Screen):
    pass

class Home(Screen):
    pass

class Discover(Screen):
    pass

class Settings(Screen):
    pass

class CustomDropDown(BoxLayout):
    pass

Builder.load_file("screens.kv")

sm = ScreenManager(transition=NoTransition())
sm.add_widget(BootUp(name='boot_up'))
sm.add_widget(SignUp(name='sign_up'))
sm.add_widget(SignIn(name='sign_in'))
sm.add_widget(MovieInput(name='movie_input'))
sm.add_widget(MovieInput2(name='movie_input2'))
sm.add_widget(Waiting(name='waiting'))

sm.add_widget(HowTo(name='how_to'))
sm.add_widget(Home(name='home'))
sm.add_widget(Discover(name='discover'))
sm.add_widget(Settings(name='settings'))

class MovieApp(App):
    def build(self):
        return sm

if __name__ == '__main__':
    MovieApp().run()

#return Image(source='Movie Placeholder Image.png')
#return Label(text='[color=000000]Hello[/color] [color=000000]World[/color]', markup = True)