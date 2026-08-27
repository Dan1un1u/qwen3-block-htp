#ifndef QWEN3_BLOCK_HTP_HOST_SESSION_H
#define QWEN3_BLOCK_HTP_HOST_SESSION_H

#include <remote.h>

struct qbh_session {
    remote_handle64 handle;
    int prepared;
};

int qbh_session_open(struct qbh_session *session);
int qbh_session_prepare(struct qbh_session *session);
int qbh_session_release(struct qbh_session *session);
int qbh_session_close(struct qbh_session *session);

#endif
