(add-to-list 'org-structure-template-alist
'("sp" . "src python :session (print pythonShell)"))
(add-to-list 'org-structure-template-alist
'("se" . "src elisp"))
(setq org-confirm-babel-evaluate nil)
(define-key org-mode-map (kbd "C-c ]") 'org-ref-insert-link)
(setq org-latex-pdf-process
  '("latexmk -pdflatex='pdflatex --syntex=1 -interaction nonstopmode' -pdf -bibtex -f %f"))
;; (setq org-latex-pdf-process (list "latexmk -f -pdf -interaction=nonstopmode -output-directory=%o %f"))
(pyvenv-workon "fem4inasdev")
(require 'org-tempo)
;; Veval_blocks -> eval blocks of latex
;; Veval_blocks_run -> eval blocks to obtain results
(setq Veval_blocks "yes") ;; yes, no, no-export 
(setq Veval_blocks_run "yes")
(setq pythonShell "py1org")
;; export_blocks: code, results, both, none
(setq export_blocks  "results")  
