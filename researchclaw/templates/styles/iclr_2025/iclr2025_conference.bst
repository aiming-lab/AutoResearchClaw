%% iclr2025_conference.bst — ICLR 2025 bibliography style
%% Symlink-equivalent to iclr2026_conference.bst (same format).
%% Bundled by AutoResearchClaw for offline compilation.

ENTRY
  { author title journal booktitle year volume number pages doi url note publisher address edition eprint archiveprefix primaryclass }
  {}
  { label }

INTEGERS { output.state before.all mid.sentence after.sentence after.block }

FUNCTION {init.state.consts}
{ #0 'before.all := #1 'mid.sentence := #2 'after.sentence := #3 'after.block := }

STRINGS { s t }

FUNCTION {output.nonnull}
{ 's :=
  output.state mid.sentence =
    { ", " * write$ }
    { output.state after.block =
        { add.period$ write$ newline$ "\newblock " write$ }
        { output.state before.all = 'write$ { add.period$ " " * write$ } if$ }
      if$
      mid.sentence 'output.state :=
    }
  if$
  s
}

FUNCTION {output}
{ duplicate$ empty$ 'pop$ 'output.nonnull if$ }

FUNCTION {output.check}
{ 't := duplicate$ empty$ { pop$ "empty " t * " in " * cite$ * warning$ } 'output.nonnull if$ }

FUNCTION {fin.entry} { add.period$ write$ newline$ }

FUNCTION {new.block}
{ output.state before.all = 'skip$ { after.block 'output.state := } if$ }

FUNCTION {not} { { #0 } { #1 } if$ }
FUNCTION {and} { 'skip$ { pop$ #0 } if$ }
FUNCTION {or} { { pop$ #1 } 'skip$ if$ }
FUNCTION {field.or.null} { duplicate$ empty$ { pop$ "" } 'skip$ if$ }
FUNCTION {emphasize} { duplicate$ empty$ { pop$ "" } { "\emph{" swap$ * "}" * } if$ }

INTEGERS { nameptr namesleft numnames }

FUNCTION {format.names}
{ 's := #1 'nameptr := s num.names$ 'numnames := numnames 'namesleft :=
    { namesleft #0 > }
    { s nameptr "{ff~}{vv~}{ll}{, jj}" format.name$ 't :=
      nameptr #1 > { namesleft #1 > { ", " * t * } { numnames #2 > { "," * } 'skip$ if$ t "others" = { " et~al." * } { " and " * t * } if$ } if$ } 't if$
      nameptr #1 + 'nameptr := namesleft #1 - 'namesleft :=
    }
  while$
}

FUNCTION {format.authors} { author empty$ { "" } { author format.names } if$ }
FUNCTION {format.title} { title empty$ { "" } { title } if$ }
FUNCTION {format.btitle} { title emphasize }
FUNCTION {format.date} { year empty$ { "" } { year } if$ }
FUNCTION {format.bvolume} { volume empty$ { "" } { "volume " volume * } if$ }
FUNCTION {format.pages} { pages empty$ { "" } { "pp. " pages * } if$ }
FUNCTION {format.url} { url empty$ { "" } { "\url{" url * "}" * } if$ }

FUNCTION {output.bibitem}
{ newline$ "\bibitem{" write$ cite$ write$ "}" write$ newline$ "" before.all 'output.state := }

FUNCTION {article}
{ output.bibitem format.authors "author" output.check new.block format.title "title" output.check new.block journal emphasize "journal" output.check format.bvolume output format.pages output format.date "year" output.check format.url output fin.entry }

FUNCTION {inproceedings}
{ output.bibitem format.authors "author" output.check new.block format.title "title" output.check new.block "In " booktitle emphasize * output format.pages output format.date "year" output.check format.url output fin.entry }

FUNCTION {conference} { inproceedings }
FUNCTION {book}
{ output.bibitem format.authors "author" output.check new.block format.btitle "title" output.check publisher output format.date "year" output.check fin.entry }

FUNCTION {misc}
{ output.bibitem format.authors output new.block format.title output new.block note output format.date output format.url output fin.entry }

FUNCTION {techreport} { misc }
FUNCTION {phdthesis} { misc }
FUNCTION {mastersthesis} { misc }
FUNCTION {unpublished} { misc }
FUNCTION {default.type} { misc }

READ
FUNCTION {sortify} { purify$ "l" change.case$ }
FUNCTION {presort} { cite$ 'label := label sortify "    " * #1 entry.max$ substring$ 'sort.key$ := }
ITERATE {presort}
SORT
FUNCTION {begin.bib} { preamble$ empty$ 'skip$ { preamble$ write$ newline$ } if$ "\begin{thebibliography}{99}" write$ newline$ }
FUNCTION {end.bib} { newline$ "\end{thebibliography}" write$ newline$ }
EXECUTE {begin.bib}
EXECUTE {init.state.consts}
ITERATE {call.type$}
EXECUTE {end.bib}
