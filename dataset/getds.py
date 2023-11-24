import os, sys

from .nyu import NYUv2
from .cityscape import CustomCityScapeDS
from .celeb import CustomCeleb
from .ox import CustomOxFordPet

from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset


def get_ds_ox(args):
    train_ds = CustomOxFordPet(split = 'trainval')
    test_ds = CustomOxFordPet(split = 'test')

    extra_train_ds, valid_ds, test_ds = random_split(test_ds, [0.8, 0.1, 0.1])
    valid_ds.mode = 'test'
    test_ds.mode = 'test'
    extra_train_ds.mode = 'train'

    train_ds = ConcatDataset([train_ds, extra_train_ds])

    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pinmem, num_workers=args.wk)
    valid_dl = DataLoader(valid_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pinmem, num_workers=args.wk)
    test_dl = DataLoader(test_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pinmem, num_workers=args.wk)

    return (train_ds, valid_ds, test_ds, train_dl, valid_dl, test_dl)

def get_ds_nyu(args):
    ds = NYUv2()

    train_ds, valid_ds, test_ds = random_split(ds, [0.8, 0.1, 0.1])

    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pinmem, num_workers=args.wk)
    valid_dl = DataLoader(valid_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pinmem, num_workers=args.wk)
    test_dl = DataLoader(test_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pinmem, num_workers=args.wk)

    return (train_ds, valid_ds, test_ds, train_dl, valid_dl, test_dl)

def get_ds_celeb(args):
    train_ds = CustomCeleb(split='train')
    valid_ds = CustomCeleb(split='valid')
    test_ds = CustomCeleb(split='test')

    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pinmem, num_workers=args.wk)
    valid_dl = DataLoader(valid_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pinmem, num_workers=args.wk)
    test_dl = DataLoader(test_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pinmem, num_workers=args.wk)

    return (train_ds, valid_ds, test_ds, train_dl, valid_dl, test_dl)

def get_ds_city(args):
    if args.citi_mode == 'fine':
        train_ds = CustomCityScapeDS(split='train', mode=args.citi_mode)
        valid_ds = CustomCityScapeDS(split='val', mode=args.citi_mode)
        test_ds = CustomCityScapeDS(split='test', mode=args.citi_mode)
    elif args.citi_mode == "coarse":
        train_ds_1 = CustomCityScapeDS(split='train', mode=args.citi_mode)
        train_ds_2 = CustomCityScapeDS(split='train_extra', mode=args.citi_mode)
        train_ds = ConcatDataset([train_ds_1, train_ds_2])
        valid_ds = CustomCityScapeDS(split='val', mode=args.citi_mode)
        test_ds = CustomCityScapeDS(split='val', mode=args.citi_mode)
    
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pinmem, num_workers=args.wk)
    valid_dl = DataLoader(valid_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pinmem, num_workers=args.wk)
    test_dl = DataLoader(test_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pinmem, num_workers=args.wk)

    return (train_ds, valid_ds, test_ds, train_dl, valid_dl, test_dl)


def get_ds(args):

    ds_mapping = {
        "oxford" : get_ds_ox,
        "nyu" : get_ds_nyu,
        "celeb" : get_ds_celeb,
        "city" : get_ds_city
    }

    train_ds, valid_ds, test_ds, train_dl, valid_dl, test_dl = ds_mapping[args.ds](args)

    return (train_ds, valid_ds, test_ds, train_dl, valid_dl, test_dl)