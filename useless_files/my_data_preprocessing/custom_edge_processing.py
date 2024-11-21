def get_edges_from_edges_txt():
    with open("data/MARA_imdb_mlh/edges.txt", "r") as input_file:
        l1_MA = {}
        l1_MD = {}
        l1_AM = {}
        l1_DM = {}

        l2_MA = {}
        l2_MD = {}
        l2_AM = {}
        l2_DM = {}

        def append(line):
            if line[0] == "0":
                if line[1] == "MA":
                    if line[2] in l1_MA:
                        l1_MA[line[2]].append(line[3])
                    else:
                        l1_MA[line[2]] = [line[3]]
                    if line[3] in l1_AM:
                        l1_AM[line[3]].append(line[2])
                    else:
                        l1_AM[line[3]] = [line[2]]
                elif line[1] == "MD":
                    if line[2] in l1_MD:
                        l1_MD[line[2]].append(line[3])
                    else:
                        l1_MD[line[2]] = [line[3]]
                    if line[3] in l1_DM:
                        l1_DM[line[3]].append(line[2])
                    else:
                        l1_DM[line[3]] = [line[2]]
            elif line[0] == "1":
                if line[1] == "MA":
                    if line[2] in l2_MA:
                        l2_MA[line[2]].append(line[3])
                    else:
                        l2_MA[line[2]] = [line[3]]
                    if line[3] in l2_AM:
                        l2_AM[line[3]].append(line[2])
                    else:
                        l2_AM[line[3]] = [line[2]]
                elif line[1] == "MD":
                    if line[2] in l2_MD:
                        l2_MD[line[2]].append(line[3])
                    else:
                        l2_MD[line[2]] = [line[3]]
                    if line[3] in l2_DM:
                        l2_DM[line[3]].append(line[2])
                    else:
                        l2_DM[line[3]] = [line[2]]

        for line in input_file:
            line = line.replace("\n", "").split(" ")
            append(line)

        print("l1_MA", len(l1_MA))
        print("l1_MD", len(l1_MD))
        print("l2_MA", len(l2_MA))
        print("l2_MD", len(l2_MD))

        l1_MAM = {}
        l1_MDM = {}
        l2_MAM = {}
        l2_MDM = {}

        for ma in l1_MA.keys():
            for a in l1_MA[ma]:
                for am in l1_AM[a]:
                    if ma in l1_MAM.keys():
                        if am != ma:
                            l1_MAM[ma].append(am)
                    elif ma != am:
                        l1_MAM[ma] = [am]

        for ma in l2_MA.keys():
            for a in l2_MA[ma]:
                for am in l2_AM[a]:
                    if ma in l2_MAM.keys():
                        if am != ma:
                            l2_MAM[ma].append(am)
                    elif ma != am:
                        l2_MAM[ma] = [am]

        for md in l1_MD.keys():
            for d in l1_MD[md]:
                for dm in l1_DM[d]:
                    if md in l1_MDM.keys():
                        if dm != md:
                            l1_MDM[md].append(dm)
                    elif dm != md:
                        l1_MDM[md] = [dm]

        for md in l2_MD.keys():
            for d in l2_MD[md]:
                for dm in l2_DM[d]:
                    if md in l2_MDM.keys():
                        if dm != md:
                            l2_MDM[md].append(dm)
                    elif dm != md:
                        l2_MDM[md] = [dm]

        print("l1_MAM", len(l1_MAM))
        print("l1_MDM", len(l1_MDM))
        print("l2_MAM", len(l2_MAM))
        print("l2_MDM", len(l2_MDM))

        print(l1_MAM["6"])
        print(l1_MAM["4"])

        sum = 0
        for ma in l2_MAM.keys():
            sum += len(l2_MAM[ma])
        print(sum)
get_edges_from_edges_txt()