from dataclasses import dataclass

from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import HTML, to_formatted_text
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import HSplit, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.widgets.base import Frame
from pygments.lexers.c_cpp import CLexer
from pygments.lexers.c_cpp import CppLexer
from pygments.lexers.go import GoLexer
from pygments.lexers.javascript import JavascriptLexer, TypeScriptLexer
from pygments.lexers.jvm import JavaLexer, KotlinLexer
from pygments.lexers.php import PhpLexer
from pygments.lexers.python import PythonLexer
from pygments.lexers.ruby import RubyLexer
from pygments.lexers.rust import RustLexer


@dataclass
class ResultEntry:
    score: float
    file: str
    line: int
    text: str


def _format_input(results):
    return [ResultEntry(score, entry.get('file'), entry.get('line'), entry.get('text')) for (score, entry) in results]


def _syntax_highlighting(text, file):
    lexer = CLexer  # use c as default generic lexer
    if file.endswith('.py'):
        lexer = PythonLexer
    elif file.endswith('.go'):
        lexer = GoLexer
    elif file.endswith('js'):
        lexer = JavascriptLexer
    elif file.endswith('ts'):
        lexer = TypeScriptLexer
    elif file.endswith('java'):
        lexer = JavaLexer
    elif file.endswith('kt') or file.endswith('kts') or file.endswith('ktm'):
        lexer = KotlinLexer
    elif file.endswith('rb'):
        lexer = RubyLexer
    elif file.endswith('php'):
        lexer = PhpLexer
    elif file.endswith('rs'):
        lexer = RustLexer
    elif file.endswith('c') or file.endswith('h'):
        lexer = CLexer
    elif file.endswith('cpp') or file.endswith('hpp'):
        lexer = CppLexer

    pigment = PygmentsLexer(lexer, sync_from_start=True)
    lex_func = pigment.lex_document(Document(text.replace('\t', '    ')))

    lines = []
    lines.append(to_formatted_text(HTML(file), style='#7474FF'))
    lines.append([('', '\n\n')])

    for i in range(0, len(text.split('\n'))):
        lines.append(lex_func(i))
        lines.append([('', '\n')])

    return [item for sublist in lines for item in sublist]


class ResultScreen():

    def _formatted_list(self):
        lines = []
        for i, result in enumerate(self.results):
            if i == self.idx:
                lines.append(to_formatted_text(HTML(
                    f'👉 {result.score:.3f} {result.file.split("/")[-1:][0] }:{result.line}'), style='#7474FF'))
            else:
                lines.append(
                    [('', f'   {result.score:.3f} {result.file.split("/")[-1:][0] }:{result.line}')])
            lines.append([('', '\n')])

        return [item for sublist in lines for item in sublist]

    def _go_down(self):
        if self.idx < len(self.results) - 1:
            self.idx += 1
        self.snippet_content.text = self.results[self.idx].text
        self.selection_content.text = self._formatted_list()

    def _go_up(self):
        if self.idx > 0:
            self.idx -= 1
        self.snippet_content.text = self.results[self.idx].text
        self.selection_content.text = self._formatted_list()

    def __init__(self, results, query):
        self.idx = 0
        self.buffer = Buffer()
        self.results = _format_input(results)

        for r in self.results:
            r.text = _syntax_highlighting(r.text, r.file)

        self.snippet_content = FormattedTextControl(
            text=self.results[self.idx].text)
        self.selection_content = FormattedTextControl(
            text=self._formatted_list())

        self.root_container = HSplit([
            Window(content=FormattedTextControl(
                text="Results for query '{}':".format(query), show_cursor=True), height=1),
            VSplit([
                Frame(Window(content=self.selection_content),
                      width=min(40, max([len(l[1]) for l in self._formatted_list()]) + 3)),
                Frame(Window(content=self.snippet_content),
                      height=Dimension(preferred=min(30, max([len(r.text) for r in self.results]))
                                       )),
            ])])
        self.layout = Layout(self.root_container)
        self.kb = KeyBindings()

        @self.kb.add('c-c')
        def _(event):
            event.app.exit()

        @self.kb.add('q')
        def _(event):
            event.app.exit()

        @self.kb.add('down')
        def _(event):
            self._go_down()

        @self.kb.add('j')
        def _(event):
            self._go_down()

        @self.kb.add('c-n')
        def _(event):
            self._go_down()

        @self.kb.add('k')
        def _(event):
            self._go_up()

        @self.kb.add('c-p')
        def _(event):
            self._go_up()

        @self.kb.add('up')
        def _(event):
            self._go_up()

        @self.kb.add('enter')
        def _(event):
            event.app.exit(result=self.idx)

        self.app = Application(
            layout=self.layout, full_screen=False, erase_when_done=True, key_bindings=self.kb)

    def run(self):
        return self.app.run()
