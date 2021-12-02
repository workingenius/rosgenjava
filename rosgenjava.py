#!env python
# -*- encoding:utf-8 -*-

import re
from pathlib import Path
import ply.lex as lex
import ply.yacc as yacc


# ########### tokenizer ##############

tokens = (
    'TYPE',
    'NAME',
    'NUMBER',
    'STRING',
    'SEP',
    'CONSTVAL',  # With the leading "="
    # 'COMMENT',

    # 'NEWLINE',
    'LSQRTBR',
    'RSQRTBR',
)

t_TYPE = r'([a-zA-Z0-9_]+\/)+[a-zA-Z0-9_]+'
t_NAME = r'[a-zA-Z0-9_]+'
t_NUMBER = r'-?[0-9]*.?[0-9]+'
t_STRING = r'".*"'
t_SEP = r'---'
t_ignore_COMMENT = r'\#.*'
t_CONSTVAL = '=.*(\n|$)'

t_ignore_NEWLINE = r'\n'
t_LSQRTBR = r'\['
t_RSQRTBR = r'\]'

t_ignore = ' \t'

def t_error(t):
    print("Illegal character '%s'" % t.value[0])


# ######## Elements ############


class Elem:
    def __str__(self):
        return self.dump()

    def __iter__(self):
        """iterate over all sub elements"""
        return iter([])

    def __eq__(self, o):
        return False


class Service(Elem):
    def __init__(self, req, resp):
        self.req = req  # Message
        self.resp = resp  # Message

    def dump(self):
        return self.req.dump() + '---\n' + self.resp.dump()

    def __iter__(self):
        yield self.req
        yield self.resp

    def __eq__(self, o):
        return isinstance(o, Service) \
                and self.req == o.req \
                and self.resp == o.resp


class Message(Elem):
    def __init__(self, field_const_lst):
        for e in field_const_lst:
            assert isinstance(e, Field) or isinstance(e, Const)

        self._field_const_lst = field_const_lst
        self.field_lst = [e for e in field_const_lst if isinstance(e, Field)]
        self.const_lst = [e for e in field_const_lst if isinstance(e, Const)]

    def dump(self):
        return ''.join([e.dump() for e in self._field_const_lst])

    def __iter__(self):
        return iter(self.field_lst + self.const_lst)

    def __eq__(self, o):
        return isinstance(o, Message) \
                and self._field_const_lst == o._field_const_lst


class Field(Elem):
    def __init__(self, name, typ):
        self.name = name  # str
        self.typ = typ  # Typ

    def dump(self):
        return '{} {}\n'.format(self.typ.dump(), self.name)

    def __iter__(self):
        yield self.typ

    def __eq__(self, o):
        return isinstance(o, Field) and \
                self.name == o.name and \
                self.type == o.type
                


class Typ(Elem):
    @property
    def is_std(self):
        return False

    def dump(self):
        raise NotImplementedError

    @property
    def java_dep(self):
        return []

    def java_expr(self):
        raise NotImplementedError


class StdTyp(Typ):
    @property
    def is_std(self):
        return True

    @property
    def java_dep(self):
        return self._java_dep

    def __init__(self, typ_name, java_counterpart, java_dep=[]):
        self.typ_name = typ_name  # str
        self.java_counterpart = java_counterpart  # str
        self._java_dep = java_dep  # List of str

    def java_expr(self):
        return self.java_counterpart

    def dump(self):
        return self.typ_name

    def __eq__(self, o):
        return isinstance(o, StdTyp) \
                and self.type_name == o.type_name


class LocalTyp(Typ):
    def __init__(self, typ_name):
        self.typ_name = typ_name  # str

    def java_expr(self):
        return self.typ_name

    def dump(self):
        return self.typ_name

    def __eq__(self, o):
        return isinstance(o, LocalTyp) \
                and self.type_name == o.type_name


class PackageTyp(Typ):
    @classmethod
    def create(cls, typ_expr):
        assert '/' in typ_expr
        parts = typ_expr.split('/')
        assert len(parts) == 2
        return cls(*parts)

    def __init__(self, package_name, typ_name):
        self.package_name = package_name  # str, only one segment
        self.typ_name = typ_name  # str

    def java_expr(self):
        return self.package_name + '.' + self.typ_name

    def dump(self):
        return self.package_name + '/' + self.typ_name

    def __eq__(self, o):
        return isinstance(o, PackageTyp) \
                and self.package_name == o.package_name \
                and self.typ_name == o.typ_name


class ArrayTyp(Typ):
    def __init__(self, base_typ, length=None):
        assert length is None or isinstance(length, int)
        self.base_typ = base_typ
        self.length = length 

    @property
    def is_std(self):
        return self.base_typ.is_std

    @property
    def java_dep(self):
        return self.base_typ.java_dep + ['java.util.List']

    def java_expr(self):
        return 'List<' + self.base_typ.java_expr() + '>'

    def dump(self):
        num = str(self.length) if self.length is not None else ''
        return self.base_typ.dump() + '[' + num + ']'

    def __iter__(self):
        yield self.base_typ

    def __eq__(self, o):
        return isinstance(o, ArrayTyp) \
                and self.base_typ == o.base_typ \
                and self.length == o.length


class Const(Elem):
    def __init__(self, name, typ, val):
        self.name = name  # str
        self.typ = typ  # Typ
        self.val = val  # int | float | str

    def dump(self):
        # string must use double quotes
        val_lit = str(self.val)  # TODO: what if number consts ?
        return '{} {}={}\n'.format(self.typ.dump(), self.name, val_lit)

    def __iter__(self):
        yield self.typ

    def __eq__(self, o):
        return isinstance(o, Const) \
                and self.name == o.name \
                and self.typ == o.type \
                and self.val == o.val


def collect_java_dep(
        elem  # Elem
    ):  # return set of import string

    def iter_java_dep(elem):
        if isinstance(elem, Typ):
            for dep in elem.java_dep:
                yield dep
        else:
            for sub in elem:
                for dep in iter_java_dep(sub):
                    yield dep

    return set(iter_java_dep(elem))


# standard types

TYPE_BOOL = StdTyp('bool', 'boolean')
TYPE_INT8 = StdTyp('int8', 'int')
TYPE_UINT8 = StdTyp('uint8', 'int')
TYPE_INT16 = StdTyp('int16', 'int')
TYPE_UINT16 = StdTyp('uint16', 'int')
TYPE_INT32 = StdTyp('int32', 'int')
TYPE_UINT32 = StdTyp('uint32', 'int')
TYPE_INT64 = StdTyp('int64', 'long')
TYPE_UINT64 = StdTyp('uint64', 'long')
TYPE_FLOAT32 = StdTyp('float32', 'float')
TYPE_FLOAT64 = StdTyp('float64', 'double')
TYPE_STRING = StdTyp('string', 'String')
TYPE_TIME = StdTyp('time', 'Time', ['org.ros.message.Time'])
TYPE_DURATION = StdTyp('duration', 'Duration', ['org.ros.message.Duration'])

STD_TYPES = [
    TYPE_BOOL,
    TYPE_INT8,
    TYPE_UINT8,
    TYPE_INT16,
    TYPE_UINT16,
    TYPE_INT32,
    TYPE_UINT32,
    TYPE_INT64,
    TYPE_UINT64,
    TYPE_FLOAT32,
    TYPE_FLOAT64,
    TYPE_STRING,
    TYPE_TIME,
    TYPE_DURATION,
] 

# ######## parser ###############


def p_service_def(p):
    """service_def : message_def SEP message_def"""
    p[0] = Service(p[1], p[3])


def p_message_def(p):
    """message_def : message_sent_lst"""
    sent_lst = p[1]
    p[0] = Message(sent_lst)


def p_empty(p):
    """empty : """
    pass


def p_message_sent_lst(p):
    """message_sent_lst : message_sent message_sent_lst
                        | empty """
    if p[1] is None:
        p[0] = []
    if len(p) == 3:
        p[0] = [p[1]] + p[2]


def p_message_sent(p):
    """message_sent : field_def
                    | const_def"""
    sent = p[1]
    p[0] = sent
                    

def p_field_def(p):
    """field_def : type_expr NAME""" 
    p[0] = Field(name=p[2], typ=p[1])


def p_type_expr(p):
    """type_expr : type_name
                 | type_name LSQRTBR RSQRTBR
                 | type_name LSQRTBR NUMBER RSQRTBR """
    base_typ = p[1]

    if len(p) == 2:
        p[0] = base_typ
    elif len(p) == 4:
        p[0] = ArrayTyp(base_typ)
    elif len(p) == 5:
        p[0] = ArrayTyp(base_typ, int(p[3]))


def p_type_name(p):
    """type_name : TYPE
                 | NAME"""
    tn = p[1]
    for typ in STD_TYPES:
        if typ.typ_name == tn:
            p[0] = typ
            return
    if '/' in tn:
        p[0] = PackageTyp.create(tn)
    else:
        p[0] = LocalTyp(tn)


def p_const_def(p):
    """const_def : type_expr NAME CONSTVAL"""
    name = p[2]
    typ = p[1]

    val_lit = p[3]
    if val_lit.startswith('='):
        val_lit = val_lit[1:]
    val_lit = val_lit.strip('\n')
    val_lit = val_lit.strip(' ')
    val_lit = val_lit.strip('\b')

    p[0] = Const(name, typ, val_lit)


def p_const_value(p):
    """const_value : NUMBER
                   | STRING
                   | NAME """
    raw = p[1]
    try:
        p[0] = int(raw)
        return
    except (ValueError, TypeError):
        pass

    try:
        p[0] = float(raw)
        return
    except (ValueError, TypeError):
        pass

    if isinstance(raw, str) and raw.startswith('"') and raw.endswith('"'):
        p[0] = eval(raw)
        return

    p[0] = str(raw)


def p_error(p):
    print('Error!')


# ########## Generate java code ######


class GeneratedFile:
    def __init__(self, filename, content):
        self.filename = filename  # str: a name, not a path
        self.content = content  # str

    def package_at(self, package_name):
        """Add package line to file content"""
        assert '.' not in package_name
        assert '/' not in package_name

        return GeneratedFile(
            self.filename,
            'package {};\n'.format(package_name) + self.content
        )

    def __str__(self):
        return '// {}:\n{}\n'.format(self.filename, self.content)


def gen_java_service(
        package_name,  # str
        service_name,  # str
        service  # Service
        ):  # retrun List of GeneratedFile

    service_name = camel_case(service_name)

    req_file = gen_java_message(
            package_name, 
            service_name + 'Request',
            service.req)

    resp_file = gen_java_message(
            package_name,
            service_name + 'Response',
            service.resp)

    fmt = """
import org.ros.internal.message.Message;

public interface {service_name} extends Message {{
    String _TYPE = {service_path};
    String _DEFINITION = {definition};
}}"""

    content = fmt.format(
        service_name=service_name,
        service_path=repr_double_quote(
            package_name + '/' + service_name),
        definition=repr_double_quote(service.dump())
    )

    return req_file + resp_file + [
        GeneratedFile(
            service_name + ".java",
            content,
        ) \
        .package_at(package_name)
    ]


def gen_java_message(
        package_name,  # str
        message_name,  # str
        message  # Message
        ):  # return List of GeneratedFile

    def gen_dep():
        return '\n'.join(
            'import {};'.format(dep)
            for dep in sorted(collect_java_dep(message))
        )

    def gen_consts():
        res = ''
        fmt = "{type_expr} {name} = {val};\n"
        for const in message.const_lst:
            res += fmt.format(
                    type_expr=const.typ.java_expr(),
                    name=const.name,
                    val=repr2(const.val))
        return res

    def gen_methods():
        res = ''

        fmt = """{type_expr} get{field_name}();
void set{field_name}({type_expr} var);
"""
        for field in message.field_lst:
            res += fmt.format(
                    type_expr=field.typ.java_expr(),
                    field_name=camel_case(field.name))

        return res
        

    fmt = """
import org.ros.internal.message.Message;
{dep}

public interface {class_name} extends Message {{
    String _TYPE = {type_expr};
    String _DEFINITION = {definition};
{consts}
{methods}
}}"""

    content = fmt.format(
        dep=gen_dep(),
        class_name=message_name,
        type_expr=repr_double_quote(
            package_name + '/' + camel_case(message_name)),  # TODO extract 
        definition=repr_double_quote(message.dump()),
        consts=indent(gen_consts(), 1),
        methods=indent(gen_methods(), 1)
    )

    return [GeneratedFile(
            camel_case(message_name) + '.java',
            content) \
        .package_at(package_name)]


# ############# Cli ##################


def process_file(the_path, args):
    is_srv = True

    if str(the_path).endswith('.srv'):
        is_srv = True
    elif str(the_path).endswith('.msg'):
        is_srv = False
    else:
        raise ValueError('Invalid file suffix')

    # Search its parent dir to get package name
    package_name = Path(the_path).absolute().parent.name
    message_name = Path(the_path).absolute().name[:-4]
    parent_path = Path(the_path).absolute().parent

    if args.tokenize:
        lexer = lex.lex()
        with open(the_path) as f:
            lexer.input(f.read())
            while True:
                tok = lexer.token()
                if not tok:
                    break
                print(tok)
        return

    lexer = lex.lex()
    if is_srv:
        parser = yacc.yacc()
    else:
        parser = yacc.yacc(start='message_def')

    with open(the_path) as f:
        ast = parser.parse(f.read())

        if args.analyze:
            print(ast)

        else:
            if is_srv:
                assert isinstance(ast, Service)
                res = gen_java_service(package_name, message_name, ast)

            else:
                assert isinstance(ast, Message)
                res = gen_java_message(package_name, message_name, ast)

            if args.show:
                print(('=' * 20 + '\n').join([str(gf) for gf in res]))

            else:
                for gf in res:
                    target_path = parent_path / gf.filename
                    print('Generating {}'.format(str(target_path)))
                    with open(target_path, 'w') as wf:
                        wf.write(gf.content)


def main():
    import os
    import argparse

    argparser = argparse.ArgumentParser(description='Process .srv/.msg file')
    argparser.add_argument('-t', '--tokenize', action='store_true', help='Run tokenizer only')
    argparser.add_argument('-a', '--analyze', action='store_true', help='Run syntax analyze only')
    argparser.add_argument('-s', '--show', action='store_true', help='Print generated java code only')
    argparser.add_argument('-r', '--recursive', action='store_true', help='Walk beneath the path and process all .srv/.msg')
    argparser.add_argument('filename', nargs=1)
    args = argparser.parse_args()

    the_path = args.filename[0]
    if args.recursive:
        for root, dirs, files in os.walk(the_path):
            for fil in files:
                if is_caring_file(fil):
                    process_file(Path(root) / fil, args)
    else:
        process_file(the_path, args)


# #### Utils ####

def _is_underscored(s):
    return re.match('^[a-z0-9_]*$', s)


def _is_big_camel_case(s):
    return re.match('^[A-Za-z0-9]*$', s) \
        and (len(s) <= 0 or s[0].isupper())


def _is_small_camel_case(s):
    return re.match('^[A-Za-z0-9]*$', s) \
        and (len(s) <= 0 or s[0].islower())


def _decapitalize(s):
    if s:
        return s[0].lower() + s[1:]
    return s


def _capitalize(s):
    if s:
        return s[0].upper() + s[1:]
    return s


def camel_case(s, big=True):
    if not s:
        return s

    if _is_underscored(s):
        ss = s.title().replace('_', '')
        if big:
            return _capitalize(ss)
        else:
            return _decapitalize(ss)

    if _is_big_camel_case(s):
        if big:
            return s
        else:
            return _decapitalize(s)

    if _is_small_camel_case(s):
        if big:
            return _capitalize(s)
        else:
            return s

    if big:
        return _capitalize(s)
    else:
        return _decapitalize(s)


def repr_double_quote(s):
    """get repr of a string with double quotes"""
    r = repr(s)
    if r.startswith("'"):
        r = r[1:-1]
        r = r.replace('"', r'\"')
        r = '"' + r + '"'
    return r


def repr2(o):
    if isinstance(o, str):
        return repr_double_quote(o)
    else:
        return repr(o)


def indent(s, ind):
    return '\n'.join(
        '    ' * ind + line for line in s.split('\n')
    )


def is_caring_file(s):
    # str -> bool
    return s.endswith('.srv') or s.endswith('.msg')


if __name__ == '__main__':
    main()

