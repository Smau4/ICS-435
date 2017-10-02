CREATE TABLE IF NOT EXISTS TrialResults (
                                        result_id integer PRIMARY KEY,
                                        trial_id integer NOT NULL,
                                        true_neg integer NOT NULL,
                                        false_neg integer NOT NULL,
                                        true_pos integer NOT NULL,
                                        false_pos integer NOT NULL,
                                        nu_val real NOT NULL,
                                        FOREIGN KEY (trial_id) REFERENCES Trials (trial_id)
                                    );