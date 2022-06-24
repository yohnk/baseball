import pull_data, clean_data, create_docs, create_tar

if __name__ == "__main__":
    pull_data.main()
    clean_data.main()
    create_docs.main()
    create_tar.main()

