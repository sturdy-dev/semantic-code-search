
from InquirerPy.separator import Separator
from InquirerPy.base.control import Choice
from InquirerPy import inquirer
import os


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 - 1)
    result[0::2] = lst
    return result


def present_results(results, query, root):
    return inquirer.select(
        message='Go to result for query "{}"'.format(query),
        border=True,
        cycle=False,
        vi_mode=True,
        qmark='',
        amark='',
        pointer='👉  ',
        height='100%',
        raise_keyboard_interrupt=False,
        mandatory=False,
        choices=intersperse(
            [
                Choice((r[1]['file'], r[1]['line']+1),
                       name='{:.3f}'.format(r[0]) + ' ' + r[1]['file'].removeprefix(root + '/') + ':' + str(r[1]['line']) + '\n\n' + r[1]['text'].replace('\t', '  ')+'\n')
                for r in results],
            Separator()
        ) + [Choice(None, name='')],
        default=(results[0][1]['file'], results[0][1]['line']+1),
    ).execute()


def open_in_editor(file, line, editor):
    if editor == 'vim':
        os.system('vim +{} {}'.format(line, file))
    elif editor == 'vscode':
        os.system('code --goto {}:{}'.format(file, line))
