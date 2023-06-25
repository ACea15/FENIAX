def flatten_list(lis):
    l=list(lis)
    i=0
#while type(max(l)) is list:

    while i < len(l):
         if type(l[i]) is tuple:
          l[i] = list(l[i])
         if type(l[i]) is list:
          if len(l[i])==0:
           del l[i]
           continue
          for j in range(len(l[i])):
            l.insert(i+j,l[i+j][j])

          del l[i+j+1]
         else:
          i=i+1

    return l
